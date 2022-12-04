rule draw_effspin_samples:
    input: 
        "src/data/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json",
        "src/data/bsplines_64m1_18q_ind18mag_ind18tilt_pl18z_posterior_samples.h5", 
        "src/data/bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_posterior_samples.h5"
    output:
        "src/data/chi_eff_chi_p_ppds.h5"
    conda:
        "environment.yml"
    script:
        "src/scripts/create_effspin_from_compspin.py"

rule update_macros:
    input:
        "src/data/bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_posterior_samples.h5",
        "src/data/bsplines_64m1_18q_ind18mag_ind18tilt_pl18z_posterior_samples.h5",
        "src/data/bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_ppds.h5",
        "src/data/bsplines_64m1_18q_ind18mag_ind18tilt_pl18z_ppds.h5",
        "src/data/chi_eff_ppds.h5",
        "src/data/gaussian-spin-xeff-xp-ppd-data.json",
        "src/data/spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5",
        "src/data/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json",
        "src/data/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5"
    output:
        "src/tex/macros.tex", 
        "src/data/macros.json"
    conda:
        "environment.yml"
    script:
        "src/scripts/create_macros.py"