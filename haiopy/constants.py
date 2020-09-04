import numpy as np


def constants(temperature=20, humidity=0.5, atmospheric_pressure=101325):
    """Commonly used constants.

    Parameters
    ----------
    temperature : float
        The temperature in degrees Celsius.
    humidity : float, [0, 1]
        The relative air humidity in the interval [0, 1].
    atmospheric_pressure
        The atmospheric pressure in Pascals.

    Returns
    -------
    constants : dict
        A dictionary with the corresponding constants.

    """

    T_kelvin_zero_c = 273.15
    T_c = temperature
    rel_hum = humidity
    T = T_c + T_kelvin_zero_c
    T_ref = 20 + T_kelvin_zero_c
    p_ref = atmospheric_pressure

    V = -6.8346*(T_ref / T)**1.261 + 4.6151

    # saturation vapor pressure
    p_sat = p_ref*10**V

    # molar concentration of water vapor in percent
    h = 100*rel_hum*p_sat/atmospheric_pressure

    # molar mass of dry air
    M_r = 0.0289644

    # molar gas constant for air
    R_mol = 8.31

    # gas constant for dry air [J/(kg*K)]
    R_l = R_mol/M_r

    # gas constant of water vapor
    R_d = 461

    # gas constant for air with relative humidity phi [J/(kg K)]
    R_f = R_l/(1-(h/100)*(1-R_l/R_d))

    # heat capacity ratio
    kappa = 1.4

    # heat conductivity
    nu = 0.0261

    # specific heat capacity
    C_v = 718

    # air density
    rho_0 = atmospheric_pressure / (R_f*T)

    # air viscosity (at 273K)
    eta = 17.1*1e-6

    # - reference pressure for SPL
    p_b = 2e-5

    c = np.sqrt(kappa*R_f*T)

    return {'c': c}


def speed_of_sound(temperature=20, humidity=0.5):
    """Calculate the speed of sound for a given temperature and air humidity.

    Parameters
    ----------
    temperature : float
        The temperature in degrees Celsius.
    humidity : float, [0, 1]
        The relative air humidity in the interval [0, 1].

    Returns
    -------
    speed_of_sound : float
        The speed of sound in meters per second.

    """
    const = constants(temperature=temperature, humidity=humidity)

    return const['c']
