"""Comprobaciones analíticas del modelo de disco (src/physics/disk.py).

Cada test contrasta contra un valor conocido en forma cerrada, no contra una
salida guardada. Si alguno falla, la física está mal, no el test.
"""
import numpy as np
import pytest

from physics import disk as D


# --------------------------------------------------------------- cinemática
def test_orbital_velocity_at_isco_is_exactly_half_c():
    # v = (1/sqrt(2x)) / sqrt(1 - 1/x);  en x=3 -> (1/sqrt6)/sqrt(2/3) = 1/2
    assert D.orbital_velocity(D.X_ISCO) == pytest.approx(0.5, rel=1e-12)


def test_orbital_velocity_reaches_c_at_photon_sphere():
    # v = (1/sqrt3)/sqrt(1 - 2/3) = 1 exacto: solo la luz orbita en x = 1.5
    assert D.orbital_velocity(D.X_PHOTON) == pytest.approx(1.0, rel=1e-12)
    assert D.orbital_velocity(1e8) < 1e-3  # newtoniano lejos


def test_omega_at_isco():
    assert D.omega_rs(3.0) == pytest.approx(1.0 / np.sqrt(54.0), rel=1e-12)


def test_time_dilation_vanishes_at_photon_sphere():
    assert D.time_dilation(D.X_PHOTON) == pytest.approx(0.0, abs=1e-12)
    assert D.time_dilation(D.X_ISCO) == pytest.approx(np.sqrt(0.5), rel=1e-12)
    assert D.time_dilation(1e8) == pytest.approx(1.0, rel=1e-7)


# --------------------------------------------------------------- redshift
def test_g_tends_to_one_far_away():
    assert D.redshift_g(1e9, 0.0) == pytest.approx(1.0, rel=1e-8)


def test_g_asymmetry_is_doppler():
    """El lado que se acerca sale corrido al azul respecto al que se aleja."""
    x = 5.0
    xi = D.image_to_bz(x, np.deg2rad(80.0))
    assert D.redshift_g(x, xi) > D.redshift_g(x, -xi)


def test_face_on_disk_has_no_doppler_asymmetry():
    """De cara (inc=0) b_z se anula y la imagen es axialmente simétrica."""
    assert D.image_to_bz(7.0, 0.0) == pytest.approx(0.0, abs=1e-15)
    assert D.redshift_g(7.0, D.image_to_bz(7.0, 0.0)) == pytest.approx(
        D.time_dilation(7.0), rel=1e-12)


def test_beaming_is_fourth_power():
    assert D.beaming_factor(2.0) == pytest.approx(16.0)


# --------------------------------------------------------------- temperatura
def test_temperature_zero_at_inner_edge():
    assert D.temperature_profile(D.X_ISCO) == pytest.approx(0.0, abs=1e-12)


def test_temperature_peaks_at_49_over_36_x_in():
    x_in = D.X_ISCO
    x = np.linspace(x_in, 40.0, 400_000)
    assert x[np.argmax(D.temperature_profile(x, x_in))] == pytest.approx(
        (49.0 / 36.0) * x_in, rel=1e-3)
    assert D.temperature_profile((49.0 / 36.0) * x_in, x_in) == pytest.approx(1.0, rel=1e-9)


def test_temperature_falls_as_r_to_minus_three_quarters():
    """Lejos del borde interno el perfil tiende a x^(-3/4)."""
    # Hay que ir muy lejos: el factor (1 - sqrt(x_in/x)) tarda en saturar.
    ratio = D.temperature_profile(4e6) / D.temperature_profile(1e6)
    assert ratio == pytest.approx(4.0**-0.75, rel=1e-3)


def test_supermassive_disks_are_cooler_at_eddington():
    """Con Mdot ∝ M queda T ∝ M^(-1/4): Sgr A* mucho más frío que un estelar."""
    t_star = D.temperature_scale_K(10.0, D.eddington_mdot(10.0))
    t_smbh = D.temperature_scale_K(4.3e6, D.eddington_mdot(4.3e6))
    assert t_star > t_smbh
    assert t_smbh / t_star == pytest.approx((4.3e6 / 10.0) ** -0.25, rel=1e-6)


def test_stellar_disk_peaks_in_xrays():
    """Orden de magnitud: 10 masas solares a Eddington -> ~10^7 K."""
    t = D.temperature_K(D.X_ISCO * 49 / 36, 10.0, D.eddington_mdot(10.0))
    assert 1e6 < t < 1e8


# --------------------------------------------------------------- geometría
@pytest.mark.parametrize("alpha", [0.0, 0.7, 2.0, np.pi, 4.5])
def test_face_on_first_crossing_is_ninety_degrees(alpha):
    assert D.first_crossing_phi(alpha, 0.0) == pytest.approx(np.pi / 2, rel=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 1.0, np.pi])
def test_edge_on_first_crossing_is_half_turn(alpha):
    assert D.first_crossing_phi(alpha, np.pi / 2) == pytest.approx(np.pi, rel=1e-12)


@pytest.mark.parametrize("inc", [0.05, 0.4, 1.0, 1.4, np.pi / 2])
@pytest.mark.parametrize("alpha", [0.0, 0.9, 2.5, 5.0])
def test_crossing_satisfies_the_geometric_condition(inc, alpha):
    """cos(phi)cos(i) + sin(phi)sin(i)cos(alpha) = 0 por construcción."""
    phi = D.first_crossing_phi(alpha, inc)
    lhs = np.cos(phi) * np.cos(inc) + np.sin(phi) * np.sin(inc) * np.cos(alpha)
    assert lhs == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < phi <= np.pi


def test_higher_order_images_are_half_turns_apart():
    ang = D.crossing_angles(1.2, 0.8, n_images=4)
    assert np.allclose(np.diff(ang.ravel()), np.pi)


def test_disk_hit_window():
    assert D.disk_hit(5.0, 3.0, 20.0)
    assert not D.disk_hit(2.0, 3.0, 20.0)
    assert not D.disk_hit(25.0, 3.0, 20.0)
    assert not D.disk_hit(np.nan, 3.0, 20.0)
