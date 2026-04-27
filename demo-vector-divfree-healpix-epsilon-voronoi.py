import gc
import importlib.util
import sys
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree


OUTDIR = Path('temp-figures')

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'axes.grid': True,
    'grid.alpha': 0.18,
    'figure.dpi': 150,
    'savefig.dpi': 240,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


def import_local_graphgp():
    spec = importlib.util.find_spec('graphgp')
    if spec is not None:
        import graphgp as gp  # type: ignore
        return gp

    here = Path(__file__).resolve().parent
    init_path = here / '__init__.py'
    spec = importlib.util.spec_from_file_location(
        'graphgp',
        init_path,
        submodule_search_locations=[str(here)],
    )
    if spec is None or spec.loader is None:
        raise ImportError('Could not load local graphgp package.')
    module = importlib.util.module_from_spec(spec)
    sys.modules['graphgp'] = module
    spec.loader.exec_module(module)
    return module


gp = import_local_graphgp()


def make_div_free_rbf_covariance(ell, sigma2=1.0, periodic=False):
    ell2 = ell * ell

    def cov_elem(x, y):
        r = y - x
        if periodic:
            r = r - jnp.round(r)
        rr2 = jnp.dot(r, r)
        base = jnp.exp(-0.5 * rr2 / ell2)
        I = jnp.eye(3, dtype=base.dtype)
        a = (1.0 - 0.5 * rr2 / ell2) * base
        b = (0.5 / ell2) * base
        return sigma2 * (a * I + b * jnp.outer(r, r))

    return cov_elem


def healpix_ring_angles(nside: int):
    if nside < 1:
        raise ValueError('nside must be >= 1')

    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    nl2 = 2 * nside
    nl4 = 4 * nside
    fact1 = 2.0 / (3.0 * nside)
    fact2 = 1.0 / (3.0 * nside * nside)

    theta = np.empty(npix, dtype=np.float64)
    phi = np.empty(npix, dtype=np.float64)

    for ipix in range(npix):
        ipix1 = ipix + 1
        if ipix1 <= ncap:
            iring = int(0.5 * (1.0 + np.sqrt(2.0 * ipix1 - 1.0)))
            iphi = ipix1 - 2 * iring * (iring - 1)
            z = 1.0 - (iring * iring) * fact2
            ang = (iphi - 0.5) * np.pi / (2.0 * iring)
        elif ipix1 <= npix - ncap:
            ip = ipix1 - ncap - 1
            iring = ip // nl4 + nside
            iphi = ip % nl4 + 1
            fodd = 0.5 * (1 + ((iring + nside) & 1))
            z = (nl2 - iring) * fact1
            ang = (iphi - fodd) * np.pi / (2.0 * nside)
        else:
            ip = npix - ipix1 + 1
            iring = int(0.5 * (1.0 + np.sqrt(2.0 * ip - 1.0)))
            iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
            z = -1.0 + (iring * iring) * fact2
            ang = (iphi - 0.5) * np.pi / (2.0 * iring)

        theta[ipix] = np.arccos(np.clip(z, -1.0, 1.0))
        phi[ipix] = np.mod(ang, 2.0 * np.pi)

    return theta, phi


def spherical_basis(theta, phi):
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)

    e_r = np.stack([st * cp, st * sp, ct], axis=-1)
    e_theta = np.stack([ct * cp, ct * sp, -st], axis=-1)
    e_phi = np.stack([-sp, cp, np.zeros_like(phi)], axis=-1)
    return e_r, e_theta, e_phi


def build_healpix_logr_grid(nside: int, n_shells: int, r_min: float, r_max: float):
    theta, phi = healpix_ring_angles(nside)
    radii = np.geomspace(r_min, r_max, n_shells)
    e_r, e_theta, e_phi = spherical_basis(theta, phi)
    unit_xyz = e_r

    points = []
    shell_ids = []
    pixel_ids = []
    for ishell, r in enumerate(radii):
        xyz = r * unit_xyz
        points.append(xyz)
        shell_ids.append(np.full(unit_xyz.shape[0], ishell, dtype=np.int32))
        pixel_ids.append(np.arange(unit_xyz.shape[0], dtype=np.int32))

    points = np.concatenate(points, axis=0)
    shell_ids = np.concatenate(shell_ids)
    pixel_ids = np.concatenate(pixel_ids)
    return {
        'points': points,
        'shell_ids': shell_ids,
        'pixel_ids': pixel_ids,
        'radii': radii,
        'theta_pix': theta,
        'phi_pix': phi,
        'unit_xyz': unit_xyz,
        'e_r_pix': e_r,
        'e_theta_pix': e_theta,
        'e_phi_pix': e_phi,
        'npix': unit_xyz.shape[0],
    }


def local_linear_divergence(points, field, k_neighbors=24, reg=1e-8):
    tree = cKDTree(points)
    dists, idx = tree.query(points, k=k_neighbors + 1)
    dists = np.asarray(dists[:, 1:])
    idx = np.asarray(idx[:, 1:])

    npts = points.shape[0]
    div = np.empty(npts, dtype=np.float64)
    cond = np.empty(npts, dtype=np.float64)
    fit_rms = np.empty(npts, dtype=np.float64)

    for i in range(npts):
        neigh = idx[i]
        X = points[neigh] - points[i]
        Y = field[neigh] - field[i]
        scale = np.median(dists[i])
        scale = max(scale, 1e-12)
        w = np.exp(-0.5 * (dists[i] / scale) ** 2)
        WX = X * w[:, None]
        A = X.T @ WX + reg * np.eye(3)
        C = X.T @ (Y * w[:, None])
        J = np.linalg.solve(A, C)
        resid = X @ J - Y
        div[i] = np.trace(J)
        cond[i] = np.linalg.cond(A)
        fit_rms[i] = np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))

    return {
        'div': div,
        'cond': cond,
        'fit_rms': fit_rms,
        'neighbor_k': k_neighbors,
    }


def shell_interface_radii(radii):
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim != 1 or radii.size < 2:
        raise ValueError('radii must be a 1D array with at least two shells')
    log_r = np.log(radii)
    face_log_r = np.empty(radii.size + 1, dtype=np.float64)
    face_log_r[1:-1] = 0.5 * (log_r[:-1] + log_r[1:])
    face_log_r[0] = log_r[0] - 0.5 * (log_r[1] - log_r[0])
    face_log_r[-1] = log_r[-1] + 0.5 * (log_r[-1] - log_r[-2])
    return np.exp(face_log_r)


def build_spherical_fvm_geometry(unit_xyz):
    def vertex_coord_key(v, ndigits=12):
        return tuple(np.round(v, decimals=ndigits))

    def dedupe_region(region, vertices, tol=1e-12):
        cleaned = []
        for idx in region:
            if not cleaned or np.linalg.norm(vertices[idx] - vertices[cleaned[-1]]) > tol:
                cleaned.append(idx)
        if len(cleaned) > 1 and np.linalg.norm(vertices[cleaned[0]] - vertices[cleaned[-1]]) <= tol:
            cleaned.pop()
        return cleaned

    sv = SphericalVoronoi(unit_xyz, radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()
    vertices = np.asarray(sv.vertices, dtype=np.float64)
    areas = np.asarray(sv.calculate_areas(), dtype=np.float64)

    edge_to_cells = {}
    cell_edges = []
    cleaned_regions = []
    for i, region in enumerate(sv.regions):
        region = dedupe_region(region, vertices)
        if len(region) < 3:
            raise RuntimeError(f'Voronoi region {i} became degenerate after deduplication.')
        cleaned_regions.append(region)

        p = unit_xyz[i]
        edges_i = []
        m = len(region)
        for k in range(m):
            a = region[k]
            b = region[(k + 1) % m]
            va = vertices[a]
            vb = vertices[b]

            cross = np.cross(va, vb)
            cross_norm = np.linalg.norm(cross)
            if cross_norm <= 1e-14:
                continue

            edge_mid = va + vb
            edge_mid /= np.linalg.norm(edge_mid)

            edge_normal = cross / cross_norm
            toward_center = p - np.dot(p, edge_mid) * edge_mid
            if np.dot(edge_normal, toward_center) > 0.0:
                edge_normal = -edge_normal

            edge_length = np.arccos(np.clip(np.dot(va, vb), -1.0, 1.0))
            key_a = vertex_coord_key(va)
            key_b = vertex_coord_key(vb)
            key = (key_a, key_b) if key_a < key_b else (key_b, key_a)

            edges_i.append({
                'key': key,
                'normal': edge_normal,
                'length': edge_length,
            })
            edge_to_cells.setdefault(key, []).append(i)
        cell_edges.append(edges_i)

    for i, edges_i in enumerate(cell_edges):
        for edge in edges_i:
            cells = edge_to_cells[edge['key']]
            if len(cells) == 2:
                edge['neighbor'] = cells[1] if cells[0] == i else cells[0]
            else:
                edge['neighbor'] = -1

    return {
        'areas': areas,
        'cell_edges': cell_edges,
        'vertices': vertices,
        'regions': cleaned_regions,
    }


def angular_surface_divergence_fvm(B_shell, e_theta, e_phi, fvm_geom):
    Btheta = np.sum(B_shell * e_theta, axis=-1)
    Bphi = np.sum(B_shell * e_phi, axis=-1)
    F_tan_xyz = Btheta[:, None] * e_theta + Bphi[:, None] * e_phi

    areas = fvm_geom['areas']
    cell_edges = fvm_geom['cell_edges']
    surface_div = np.empty(B_shell.shape[0], dtype=np.float64)

    for i, edges_i in enumerate(cell_edges):
        flux = 0.0
        Fi = F_tan_xyz[i]
        for edge in edges_i:
            j = edge['neighbor']
            if j >= 0:
                F_edge = 0.5 * (Fi + F_tan_xyz[j])
            else:
                F_edge = Fi
            flux += edge['length'] * np.dot(F_edge, edge['normal'])
        surface_div[i] = flux / areas[i]

    return {
        'surface_div': surface_div,
        'cell_area_unit_sphere': areas,
    }


def spherical_divergence_fvm_diagnostic(B_shell, radii, e_r, e_theta, e_phi, fvm_geom):
    Br, Btheta, Bphi = xyz_to_spherical_components(B_shell, e_r, e_theta, e_phi)

    r_faces = shell_interface_radii(radii)
    shell_vol_coeff = (r_faces[1:] ** 3 - r_faces[:-1] ** 3) / 3.0

    Br_faces = np.empty((radii.size + 1, Br.shape[1]), dtype=np.float64)
    Br_faces[1:-1] = 0.5 * (Br[:-1] + Br[1:])
    Br_faces[0] = Br[0]
    Br_faces[-1] = Br[-1]

    radial_flux_out = (r_faces[1:, None] ** 2) * Br_faces[1:]
    radial_flux_in = (r_faces[:-1, None] ** 2) * Br_faces[:-1]
    radial_term = (radial_flux_out - radial_flux_in) / shell_vol_coeff[:, None]

    npix = B_shell.shape[1]
    surface_div = np.empty((B_shell.shape[0], npix), dtype=np.float64)
    for ishell in range(B_shell.shape[0]):
        res = angular_surface_divergence_fvm(B_shell[ishell], e_theta, e_phi, fvm_geom)
        surface_div[ishell] = res['surface_div']

    angular_term = surface_div / radii[:, None]
    div_sph = radial_term + angular_term

    return {
        'div': div_sph,
        'Br': Br,
        'Btheta': Btheta,
        'Bphi': Bphi,
        'r_faces': r_faces,
        'radial_term': radial_term,
        'angular_term': angular_term,
        'surface_div_unit_sphere': surface_div,
        'cell_area_unit_sphere': fvm_geom['areas'],
    }


def xyz_to_spherical_components(B_shell, e_r, e_theta, e_phi):
    Br = np.sum(B_shell * e_r[None, :, :], axis=-1)
    Btheta = np.sum(B_shell * e_theta[None, :, :], axis=-1)
    Bphi = np.sum(B_shell * e_phi[None, :, :], axis=-1)
    return Br, Btheta, Bphi


def shell_statistics(values_2d):
    mean = np.mean(values_2d, axis=1)
    rms = np.sqrt(np.mean(values_2d ** 2, axis=1))
    stderr = np.std(values_2d, axis=1) / np.sqrt(values_2d.shape[1])
    return mean, rms, stderr


def interior_mask(n_shells, npix):
    mask = np.ones((n_shells, npix), dtype=bool)
    if n_shells >= 2:
        mask[0] = False
        mask[-1] = False
    return mask


def save_summary(path, config, metrics):
    lines = []
    lines.append('HEALPix × log-r divergence diagnostics')
    lines.append('=================================')
    lines.append('Grid:')
    for key in ['nside', 'n_shells', 'npix_per_shell', 'N_total', 'r_min', 'r_max']:
        lines.append(f'  {key} = {config[key]}')
    lines.append('GP:')
    for key in ['ell', 'sigma2', 'graph_n0', 'graph_k']:
        lines.append(f'  {key} = {config[key]}')
    lines.append('Diagnostics:')
    for key, value in metrics.items():
        lines.append(f'  {key} = {value}')
    path.write_text('\n'.join(lines) + '\n')


def plot_mid_shell_maps(theta_pix, phi_pix, epsilon_primary_mid, epsilon_spherical_mid, prefix):
    phi_deg = np.degrees(phi_pix)
    theta_deg = np.degrees(theta_pix)

    vmax = np.nanpercentile(np.abs(np.concatenate([epsilon_primary_mid, epsilon_spherical_mid])), 99)
    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.0), constrained_layout=True)

    sc0 = axs[0].scatter(phi_deg, theta_deg, c=epsilon_primary_mid, s=34, vmin=-vmax, vmax=vmax)
    axs[0].set_title(r'Mid-shell local Cartesian $\epsilon$')
    axs[0].set_xlabel(r'$\phi$ [deg]')
    axs[0].set_ylabel(r'$\theta$ [deg]')

    sc1 = axs[1].scatter(phi_deg, theta_deg, c=epsilon_spherical_mid, s=34, vmin=-vmax, vmax=vmax)
    axs[1].set_title(r'Mid-shell spherical finite-volume $\epsilon$')
    axs[1].set_xlabel(r'$\phi$ [deg]')
    axs[1].set_ylabel(r'$\theta$ [deg]')

    cb = fig.colorbar(sc1, ax=axs, shrink=0.88)
    cb.set_label(r'$\epsilon = \ell (\nabla\!\cdot\!B)/B_{\rm rms}$')
    fig.savefig(OUTDIR / f'{prefix}_divergence_mid_shell.png', bbox_inches='tight')
    plt.close(fig)


def plot_meridional_slice(points, epsilon_primary, epsilon_spherical, prefix):
    phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2 * np.pi)
    dphi = np.minimum(np.abs(phi), np.abs(phi - 2 * np.pi))
    keep = dphi < np.deg2rad(12.0)
    x = points[keep, 0]
    z = points[keep, 2]

    vmax = np.nanpercentile(np.abs(np.concatenate([epsilon_primary[keep], epsilon_spherical[keep]])), 99)
    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.0), constrained_layout=True)

    sc0 = axs[0].scatter(x, z, c=epsilon_primary[keep], s=30, vmin=-vmax, vmax=vmax)
    axs[0].set_title(r'Meridional slice: local Cartesian $\epsilon$')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('z')
    axs[0].set_aspect('equal')

    sc1 = axs[1].scatter(x, z, c=epsilon_spherical[keep], s=30, vmin=-vmax, vmax=vmax)
    axs[1].set_title(r'Meridional slice: spherical finite-volume $\epsilon$')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('z')
    axs[1].set_aspect('equal')

    cb = fig.colorbar(sc1, ax=axs, shrink=0.88)
    cb.set_label(r'$\epsilon$')
    fig.savefig(OUTDIR / f'{prefix}_divergence_meridional_slice.png', bbox_inches='tight')
    plt.close(fig)


def plot_epsilon_histograms(epsilon_primary, epsilon_secondary, epsilon_spherical, interior, prefix):
    eps_all = np.concatenate([epsilon_primary.ravel(), epsilon_secondary.ravel(), epsilon_spherical.ravel()])
    lo = np.nanpercentile(eps_all, 0.5)
    hi = np.nanpercentile(eps_all, 99.5)
    bins = np.linspace(lo, hi, 70)

    fig, axs = plt.subplots(1, 2, figsize=(14.0, 5.0), constrained_layout=True)

    axs[0].hist(epsilon_primary.ravel(), bins=bins, density=True, histtype='step', linewidth=2.0, label='local xyz (all)')
    axs[0].hist(epsilon_primary[interior].ravel(), bins=bins, density=True, histtype='step', linewidth=2.0, label='local xyz (interior)')
    axs[0].axvline(0.0, ls='--', lw=1.0, color='0.45')
    axs[0].set_title(r'Irregular-grid Cartesian $\epsilon$')
    axs[0].set_xlabel(r'$\epsilon = \ell (\nabla\!\cdot\!B)/B_{\rm rms}$')
    axs[0].set_ylabel(r'probability density')
    axs[0].legend(loc='upper right')

    axs[1].hist(epsilon_spherical.ravel(), bins=bins, density=True, histtype='step', linewidth=2.0, label='spherical finite-volume (all)')
    axs[1].hist(epsilon_spherical[interior].ravel(), bins=bins, density=True, histtype='step', linewidth=2.0, label='spherical finite-volume (interior)')
    axs[1].axvline(0.0, ls='--', lw=1.0, color='0.45')
    axs[1].set_title(r'Spherical finite-volume $\epsilon$ on HEALPix $\times$ log-r')
    axs[1].set_xlabel(r'$\epsilon = \ell (\nabla\!\cdot\!B)/B_{\rm rms}$')
    axs[1].set_ylabel(r'probability density')
    axs[1].legend(loc='upper right')

    fig.savefig(OUTDIR / f'{prefix}_divergence_histograms.png', bbox_inches='tight')
    plt.close(fig)


def plot_shell_profile(radii, epsilon_primary, epsilon_secondary, epsilon_spherical, prefix):
    mean_p, rms_p, _ = shell_statistics(epsilon_primary)
    mean_s, rms_s, _ = shell_statistics(epsilon_secondary)
    mean_sp, rms_sp, _ = shell_statistics(epsilon_spherical)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(radii, mean_p, 'o-', lw=1.7, ms=4.0, label=r'local xyz mean')
    ax.plot(radii, rms_p, 'o--', lw=2.0, ms=4.0, label=r'local xyz rms')
    ax.plot(radii, mean_sp, 'd-', lw=1.5, ms=4.0, label=r'spherical finite-volume mean')
    ax.plot(radii, rms_sp, 'd--', lw=2.0, ms=4.0, label=r'spherical finite-volume rms')
    ax.axhline(0.0, ls='--', lw=1.0, color='0.45')
    ax.set_xscale('log')
    ax.set_title(r'Radial shell profile of normalized divergence')
    ax.set_xlabel(r'radius $r$')
    ax.set_ylabel(r'$\epsilon$')
    ax.legend(loc='best')
    fig.savefig(OUTDIR / f'{prefix}_divergence_shell_profile.png', bbox_inches='tight')
    plt.close(fig)


def generate_demo(
    nside=30,
    n_shells=50,
    r_min=0.08,
    r_max=1.0,
    n0=256,
    graph_k=8,
    ell=0.09,
    sigma2=1.0,
    seed=0,
    div_k_primary=24,
    div_k_secondary=48,
    angular_k=8,
):
    grid = build_healpix_logr_grid(nside=nside, n_shells=n_shells, r_min=r_min, r_max=r_max)
    points = grid['points']
    radii = grid['radii']
    npix = grid['npix']

    graph = gp.build_graph(jnp.asarray(points), n0=n0, k=graph_k)
    cov = make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=False)
    xi = jr.normal(jr.key(seed), (points.shape[0], 3))
    B = gp.generate_vector(graph, cov, xi, fast_jit=True)
    B = np.asarray(B)

    B_rms = float(np.sqrt(np.mean(np.sum(B ** 2, axis=1))))

    primary = local_linear_divergence(points, B, k_neighbors=div_k_primary)
    secondary = local_linear_divergence(points, B, k_neighbors=div_k_secondary)

    B_shell = B.reshape(n_shells, npix, 3)
    fvm_geom = build_spherical_fvm_geometry(grid['unit_xyz'])
    spherical = spherical_divergence_fvm_diagnostic(
        B_shell,
        radii,
        grid['e_r_pix'],
        grid['e_theta_pix'],
        grid['e_phi_pix'],
        fvm_geom,
    )

    div_primary = primary['div'].reshape(n_shells, npix)
    div_secondary = secondary['div'].reshape(n_shells, npix)
    div_spherical = spherical['div']

    epsilon_primary = ell * div_primary / B_rms
    epsilon_secondary = ell * div_secondary / B_rms
    epsilon_spherical = ell * div_spherical / B_rms

    interior = interior_mask(n_shells, npix)

    metrics = {
        'B_rms': f'{B_rms:.10f}',
        'primary_k': div_k_primary,
        'secondary_k': div_k_secondary,
        'spherical_operator': 'finite_volume',
        'primary_div_rms_all': f'{np.sqrt(np.mean(div_primary ** 2)):.10f}',
        'primary_div_rms_interior': f'{np.sqrt(np.mean(div_primary[interior] ** 2)):.10f}',
        'secondary_div_rms_all': f'{np.sqrt(np.mean(div_secondary ** 2)):.10f}',
        'secondary_div_rms_interior': f'{np.sqrt(np.mean(div_secondary[interior] ** 2)):.10f}',
        'spherical_div_rms_all': f'{np.sqrt(np.mean(div_spherical ** 2)):.10f}',
        'spherical_div_rms_interior': f'{np.sqrt(np.mean(div_spherical[interior] ** 2)):.10f}',
        'epsilon_primary_rms_all': f'{np.sqrt(np.mean(epsilon_primary ** 2)):.10f}',
        'epsilon_primary_rms_interior': f'{np.sqrt(np.mean(epsilon_primary[interior] ** 2)):.10f}',
        'epsilon_secondary_rms_all': f'{np.sqrt(np.mean(epsilon_secondary ** 2)):.10f}',
        'epsilon_secondary_rms_interior': f'{np.sqrt(np.mean(epsilon_secondary[interior] ** 2)):.10f}',
        'epsilon_spherical_rms_all': f'{np.sqrt(np.mean(epsilon_spherical ** 2)):.10f}',
        'epsilon_spherical_rms_interior': f'{np.sqrt(np.mean(epsilon_spherical[interior] ** 2)):.10f}',
        'primary_div_max_abs': f'{np.max(np.abs(div_primary)):.10f}',
        'secondary_div_max_abs': f'{np.max(np.abs(div_secondary)):.10f}',
        'spherical_div_max_abs': f'{np.max(np.abs(div_spherical)):.10f}',
        'median_cond_primary': f'{np.median(primary["cond"]):.10f}',
        'median_cond_secondary': f'{np.median(secondary["cond"]):.10f}',
        'median_fit_rms_primary': f'{np.median(primary["fit_rms"]):.10f}',
        'median_fit_rms_secondary': f'{np.median(secondary["fit_rms"]):.10f}',
        'mean_cell_area_unit_sphere': f'{np.mean(spherical["cell_area_unit_sphere"]):.10f}',
        'std_cell_area_unit_sphere': f'{np.std(spherical["cell_area_unit_sphere"]):.10f}',
    }

    config = {
        'nside': nside,
        'n_shells': n_shells,
        'npix_per_shell': npix,
        'N_total': points.shape[0],
        'r_min': f'{r_min:.6f}',
        'r_max': f'{r_max:.6f}',
        'ell': f'{ell:.6f}',
        'sigma2': f'{sigma2:.6f}',
        'graph_n0': n0,
        'graph_k': graph_k,
    }

    mid_shell = n_shells // 2
    plot_mid_shell_maps(grid['theta_pix'], grid['phi_pix'], epsilon_primary[mid_shell], epsilon_spherical[mid_shell], 'demo_vector_healpix_logr_v2')
    gc.collect()
    plot_meridional_slice(points, epsilon_primary.ravel(), epsilon_spherical.ravel(), 'demo_vector_healpix_logr_v2')
    gc.collect()
    plot_epsilon_histograms(epsilon_primary, epsilon_secondary, epsilon_spherical, interior, 'demo_vector_healpix_logr_v2')
    gc.collect()
    plot_shell_profile(radii, epsilon_primary, epsilon_secondary, epsilon_spherical, 'demo_vector_healpix_logr_v2')
    gc.collect()

    save_summary(OUTDIR / 'demo_vector_healpix_logr_divergence_summary_v2.txt', config, metrics)

    print('Saved outputs to', OUTDIR)
    print('B_rms =', metrics['B_rms'])
    print('epsilon_primary_rms_all =', metrics['epsilon_primary_rms_all'])
    print('epsilon_spherical_rms_all =', metrics['epsilon_spherical_rms_all'])


if __name__ == '__main__':
    generate_demo()
