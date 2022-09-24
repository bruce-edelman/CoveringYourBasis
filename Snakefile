rule draw_chieff_samples:
    input: 
        "src/data/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json",
        "src/data/mspline_50m1_16ind_compspins_smoothprior_powerlaw_q_z_posterior_samples.h5"
    output:
        "src/data/chi_eff_ppds.h5"
    conda:
        "environment.yml"
    script:
        "src/scripts/create_chieff_from_compspin.py"
