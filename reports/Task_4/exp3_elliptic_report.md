# Task 4: elliptical Kepler experiment

Execution: completed. Scientific acceptance: False.

All seeds and invalid estimates are retained. No threshold or estimator is selected after observing outcomes.

## Acceptance

| criterion                                                     | passed   | details                              |
|---------------------------------------------------------------|----------|--------------------------------------|
| All20 CUDA trainings:2000steps B256                           | True     |                                      |
| H3 M0_Koopman: one phase method passes all5 seeds             | True     | {'S3b_std': True, 'S3b_diff': True}  |
| H3 M0_Koopman: DMD degradation                                | True     |                                      |
| H3 M1_Ellip_Inv: one phase method passes all5 seeds           | True     | {'S3b_std': True, 'S3b_diff': False} |
| H3 M1_Ellip_Inv: DMD degradation                              | True     |                                      |
| Zero-shot M1_circ mean phase gap>.35                          | True     |                                      |
| M1_Ellip_Inv phase gap<.12 all5 and at least3x mean reduction | True     |                                      |
| M0_Koopman T3 fails recovery                                  | True     |                                      |
| M1_Ellip_Inv T3 mean/worst law thresholds                     | True     |                                      |
| M2_Ellip_Inv T3 mean/worst law thresholds                     | True     |                                      |
| M1Inv norm OOS loga R2<.75                                    | False    |                                      |
| M1Inv omega OOS energy R2>.94                                 | True     |                                      |

## GPU training

|   seed | model           | status    |   completed_steps | device   |   elapsed_seconds |
|--------|-----------------|-----------|-------------------|----------|-------------------|
|     42 | M1_Ellip_Short  | completed |              2000 | cuda     |           42.5183 |
|     42 | M1_Ellip_Inv    | completed |              2000 | cuda     |           81.2189 |
|     42 | M2_Ellip_Inv    | completed |              2000 | cuda     |           89.282  |
|     42 | M3_Ellip_SIGReg | completed |              2000 | cuda     |           81.6931 |
|     43 | M1_Ellip_Short  | completed |              2000 | cuda     |           42.3206 |
|     43 | M1_Ellip_Inv    | completed |              2000 | cuda     |           80.4943 |
|     43 | M2_Ellip_Inv    | completed |              2000 | cuda     |           88.6553 |
|     43 | M3_Ellip_SIGReg | completed |              2000 | cuda     |           82.9039 |
|     44 | M1_Ellip_Short  | completed |              2000 | cuda     |           43.6244 |
|     44 | M1_Ellip_Inv    | completed |              2000 | cuda     |           81.4933 |
|     44 | M2_Ellip_Inv    | completed |              2000 | cuda     |           89.639  |
|     44 | M3_Ellip_SIGReg | completed |              2000 | cuda     |           83.3284 |
|     45 | M1_Ellip_Short  | completed |              2000 | cuda     |           42.7743 |
|     45 | M1_Ellip_Inv    | completed |              2000 | cuda     |           81.6839 |
|     45 | M2_Ellip_Inv    | completed |              2000 | cuda     |           90.4081 |
|     45 | M3_Ellip_SIGReg | completed |              2000 | cuda     |           87.5132 |
|     46 | M1_Ellip_Short  | completed |              2000 | cuda     |           43.7819 |
|     46 | M1_Ellip_Inv    | completed |              2000 | cuda     |           86.74   |
|     46 | M2_Ellip_Inv    | completed |              2000 | cuda     |           89.7132 |
|     46 | M3_Ellip_SIGReg | completed |              2000 | cuda     |           83.0795 |

## Frequency estimates: all models, splits and estimators

|   seed | model           | split           | regime   | method   |         slope |   slope_abs_error |            r2 |   invalid |
|--------|-----------------|-----------------|----------|----------|---------------|-------------------|---------------|-----------|
|     42 | M0_Koopman      | test_id_ellip   | T1       | S2       |  -1.43612     |       0.0638816   |   0.993908    |         0 |
|     42 | M0_Koopman      | test_id_ellip   | T1       | S3a_diff |  -1.76519     |       0.265194    |   0.814034    |         0 |
|     42 | M0_Koopman      | test_id_ellip   | T1       | S3b_diff |  -1.50279     |       0.00279164  |   0.99944     |         0 |
|     42 | M0_Koopman      | test_id_ellip   | T1       | S3b_std  |  -1.50377     |       0.0037735   |   0.999713    |         0 |
|     42 | M0_Koopman      | test_id_ellip   | T3       | S2       |   0.0149887   |       1.51499     |   0.141471    |         0 |
|     42 | M0_Koopman      | test_id_ellip   | T3       | S3a_diff |   0.0118309   |       1.51183     |   0.0822223   |         0 |
|     42 | M0_Koopman      | test_id_ellip   | T3       | S3b_diff |   0.0116639   |       1.51166     |   0.0832691   |         0 |
|     42 | M0_Koopman      | test_id_ellip   | T3       | S3b_std  |   0.0147915   |       1.51479     |   0.128362    |         0 |
|     42 | M0_Koopman      | test_ood_ecc    | T1       | S2       |  -1.43663     |       0.0633724   |   0.993119    |         0 |
|     42 | M0_Koopman      | test_ood_ecc    | T1       | S3a_diff |  -1.51835     |       0.0183476   |   0.888423    |         0 |
|     42 | M0_Koopman      | test_ood_ecc    | T1       | S3b_diff |  -1.49714     |       0.00286081  |   0.999152    |         0 |
|     42 | M0_Koopman      | test_ood_ecc    | T1       | S3b_std  |  -1.49785     |       0.00214914  |   0.999732    |         0 |
|     42 | M0_Koopman      | test_ood_ecc    | T3       | S2       |   0.0178051   |       1.51781     |   0.259819    |         0 |
|     42 | M0_Koopman      | test_ood_ecc    | T3       | S3a_diff |   0.0174801   |       1.51748     |   0.210743    |         0 |
|     42 | M0_Koopman      | test_ood_ecc    | T3       | S3b_diff |   0.0172239   |       1.51722     |   0.215588    |         0 |
|     42 | M0_Koopman      | test_ood_ecc    | T3       | S3b_std  |   0.0186164   |       1.51862     |   0.243681    |         0 |
|     42 | M0_Koopman      | test_iso_energy | T3       | S2       |   0.0155554   |       1.51556     |   0.170219    |         0 |
|     42 | M0_Koopman      | test_iso_energy | T3       | S3a_diff |   0.0147893   |       1.51479     |   0.150869    |         0 |
|     42 | M0_Koopman      | test_iso_energy | T3       | S3b_diff |   0.0142048   |       1.5142      |   0.147796    |         0 |
|     42 | M0_Koopman      | test_iso_energy | T3       | S3b_std  |   0.0162558   |       1.51626     |   0.169245    |         0 |
|     42 | M0_Koopman      | test_ellip_peri | T3       | S2       |  -0.015855    |       1.48414     |   0.572758    |         0 |
|     42 | M0_Koopman      | test_ellip_peri | T3       | S3a_diff |  -0.0109825   |       1.48902     |   0.467432    |         0 |
|     42 | M0_Koopman      | test_ellip_peri | T3       | S3b_diff |  -0.0109123   |       1.48909     |   0.48664     |         0 |
|     42 | M0_Koopman      | test_ellip_peri | T3       | S3b_std  |  -0.0152909   |       1.48471     |   0.539341    |         0 |
|     42 | M0_Koopman      | test_ellip_apo  | T3       | S2       |   0.0227756   |       1.52278     |   0.895218    |         0 |
|     42 | M0_Koopman      | test_ellip_apo  | T3       | S3a_diff |   0.0223563   |       1.52236     |   0.947158    |         0 |
|     42 | M0_Koopman      | test_ellip_apo  | T3       | S3b_diff |   0.0207901   |       1.52079     |   0.947576    |         0 |
|     42 | M0_Koopman      | test_ellip_apo  | T3       | S3b_std  |   0.0252933   |       1.52529     |   0.913292    |         0 |
|     42 | M1_circ         | test_id_ellip   | T1       | S2       |  -1.43614     |       0.063859    |   0.99386     |         0 |
|     42 | M1_circ         | test_id_ellip   | T1       | S3a_diff |  -1.75381     |       0.253807    |   0.816547    |         0 |
|     42 | M1_circ         | test_id_ellip   | T1       | S3b_diff |  -1.50277     |       0.00277218  |   0.999425    |         0 |
|     42 | M1_circ         | test_id_ellip   | T1       | S3b_std  |  -1.50381     |       0.00381303  |   0.999705    |         0 |
|     42 | M1_circ         | test_id_ellip   | T3       | S2       |  -1.00837     |       0.49163     |   0.619445    |         0 |
|     42 | M1_circ         | test_id_ellip   | T3       | S3a_diff |  -1.20215     |       0.297847    |   0.611242    |         0 |
|     42 | M1_circ         | test_id_ellip   | T3       | S3b_diff |  -1.20285     |       0.297153    |   0.611126    |         0 |
|     42 | M1_circ         | test_id_ellip   | T3       | S3b_std  |  -1.17648     |       0.323519    |   0.619864    |         0 |
|     42 | M1_circ         | test_ood_ecc    | T1       | S2       |  -1.43717     |       0.0628279   |   0.993077    |         0 |
|     42 | M1_circ         | test_ood_ecc    | T1       | S3a_diff |  -1.50974     |       0.00974029  |   0.893081    |         0 |
|     42 | M1_circ         | test_ood_ecc    | T1       | S3b_diff |  -1.49699     |       0.00301165  |   0.99914     |         0 |
|     42 | M1_circ         | test_ood_ecc    | T1       | S3b_std  |  -1.49751     |       0.00248983  |   0.999734    |         0 |
|     42 | M1_circ         | test_ood_ecc    | T3       | S2       |  -0.855124    |       0.644876    |   0.5779      |         0 |
|     42 | M1_circ         | test_ood_ecc    | T3       | S3a_diff |  -1.42882     |       0.0711839   |   0.6919      |         0 |
|     42 | M1_circ         | test_ood_ecc    | T3       | S3b_diff |  -1.42871     |       0.0712858   |   0.691964    |         0 |
|     42 | M1_circ         | test_ood_ecc    | T3       | S3b_std  |  -1.28467     |       0.215328    |   0.679946    |         0 |
|     42 | M1_circ         | test_iso_energy | T3       | S2       |  -1.02411     |       0.475894    |   0.500987    |         0 |
|     42 | M1_circ         | test_iso_energy | T3       | S3a_diff |  -1.22849     |       0.271507    |   0.509614    |         0 |
|     42 | M1_circ         | test_iso_energy | T3       | S3b_diff |  -1.22888     |       0.271121    |   0.509567    |         0 |
|     42 | M1_circ         | test_iso_energy | T3       | S3b_std  |  -1.20164     |       0.29836     |   0.507569    |         0 |
|     42 | M1_circ         | test_ellip_peri | T3       | S2       |  -0.371048    |       1.12895     |   0.216154    |         0 |
|     42 | M1_circ         | test_ellip_peri | T3       | S3a_diff |  -0.371168    |       1.12883     |   0.216665    |         0 |
|     42 | M1_circ         | test_ellip_peri | T3       | S3b_diff |  -0.371037    |       1.12896     |   0.216591    |         0 |
|     42 | M1_circ         | test_ellip_peri | T3       | S3b_std  |  -0.370427    |       1.12957     |   0.21605     |         0 |
|     42 | M1_circ         | test_ellip_apo  | T3       | S2       |  -0.849016    |       0.650984    |   0.844591    |         0 |
|     42 | M1_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.4602      |       0.0397951   |   0.999889    |         0 |
|     42 | M1_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.46037     |       0.0396289   |   0.999888    |         0 |
|     42 | M1_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.33901     |       0.160986    |   0.98969     |         0 |
|     42 | M2_circ         | test_id_ellip   | T1       | S2       |  -1.43655     |       0.0634511   |   0.99382     |         0 |
|     42 | M2_circ         | test_id_ellip   | T1       | S3a_diff |  -1.75537     |       0.255366    |   0.827733    |         0 |
|     42 | M2_circ         | test_id_ellip   | T1       | S3b_diff |  -1.50306     |       0.00306237  |   0.999453    |         0 |
|     42 | M2_circ         | test_id_ellip   | T1       | S3b_std  |  -1.50402     |       0.00401723  |   0.999714    |         0 |
|     42 | M2_circ         | test_id_ellip   | T3       | S2       |  -1.01151     |       0.488488    |   0.636064    |         0 |
|     42 | M2_circ         | test_id_ellip   | T3       | S3a_diff |  -1.16525     |       0.33475     |   0.628005    |         0 |
|     42 | M2_circ         | test_id_ellip   | T3       | S3b_diff |  -1.17001     |       0.329989    |   0.627118    |         0 |
|     42 | M2_circ         | test_id_ellip   | T3       | S3b_std  |  -1.16733     |       0.332668    |   0.63215     |         0 |
|     42 | M2_circ         | test_ood_ecc    | T1       | S2       |  -1.4386      |       0.0614019   |   0.993019    |         0 |
|     42 | M2_circ         | test_ood_ecc    | T1       | S3a_diff |  -1.62447     |       0.124471    |   0.89953     |         0 |
|     42 | M2_circ         | test_ood_ecc    | T1       | S3b_diff |  -1.49716     |       0.0028404   |   0.999112    |         0 |
|     42 | M2_circ         | test_ood_ecc    | T1       | S3b_std  |  -1.49741     |       0.00259277  |   0.999742    |         0 |
|     42 | M2_circ         | test_ood_ecc    | T3       | S2       |  -0.953453    |       0.546547    |   0.612261    |         0 |
|     42 | M2_circ         | test_ood_ecc    | T3       | S3a_diff |  -1.4314      |       0.0686002   |   0.712359    |         0 |
|     42 | M2_circ         | test_ood_ecc    | T3       | S3b_diff |  -1.43778     |       0.0622235   |   0.712878    |         0 |
|     42 | M2_circ         | test_ood_ecc    | T3       | S3b_std  |  -1.34196     |       0.158045    |   0.706371    |         0 |
|     42 | M2_circ         | test_iso_energy | T3       | S2       |  -1.01676     |       0.483235    |   0.483131    |         0 |
|     42 | M2_circ         | test_iso_energy | T3       | S3a_diff |  -1.17762     |       0.322377    |   0.502894    |         0 |
|     42 | M2_circ         | test_iso_energy | T3       | S3b_diff |  -1.18347     |       0.31653     |   0.503081    |         0 |
|     42 | M2_circ         | test_iso_energy | T3       | S3b_std  |  -1.18529     |       0.314709    |   0.500565    |         0 |
|     42 | M2_circ         | test_ellip_peri | T3       | S2       |  -0.154918    |       1.34508     |   0.0664564   |         0 |
|     42 | M2_circ         | test_ellip_peri | T3       | S3a_diff |  -0.155925    |       1.34408     |   0.0674894   |         0 |
|     42 | M2_circ         | test_ellip_peri | T3       | S3b_diff |  -0.15557     |       1.34443     |   0.0672048   |         0 |
|     42 | M2_circ         | test_ellip_peri | T3       | S3b_std  |  -0.154608    |       1.34539     |   0.0663621   |         0 |
|     42 | M2_circ         | test_ellip_apo  | T3       | S2       |  -0.865288    |       0.634712    |   0.864759    |         0 |
|     42 | M2_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.36184     |       0.138155    |   0.999717    |         0 |
|     42 | M2_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.37125     |       0.128754    |   0.999698    |         0 |
|     42 | M2_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.30537     |       0.194629    |   0.994125    |         0 |
|     42 | M3_circ         | test_id_ellip   | T1       | S2       |  -1.43394     |       0.0660582   |   0.797366    |         0 |
|     42 | M3_circ         | test_id_ellip   | T1       | S3a_diff | nan           |     nan           | nan           |         7 |
|     42 | M3_circ         | test_id_ellip   | T1       | S3b_diff |  -1.54811     |       0.0481147   |   0.0743944   |         0 |
|     42 | M3_circ         | test_id_ellip   | T1       | S3b_std  |   0.401848    |       1.90185     |   0.00776529  |         0 |
|     42 | M3_circ         | test_id_ellip   | T3       | S2       |  -0.769119    |       0.730881    |   0.17137     |         0 |
|     42 | M3_circ         | test_id_ellip   | T3       | S3a_diff |  -0.695125    |       0.804875    |   0.153456    |         0 |
|     42 | M3_circ         | test_id_ellip   | T3       | S3b_diff |  -0.865049    |       0.634951    |   0.199203    |         0 |
|     42 | M3_circ         | test_id_ellip   | T3       | S3b_std  |  -0.743663    |       0.756337    |   0.109972    |         0 |
|     42 | M3_circ         | test_ood_ecc    | T1       | S2       |  -1.44421     |       0.0557947   |   0.993505    |         0 |
|     42 | M3_circ         | test_ood_ecc    | T1       | S3a_diff | nan           |     nan           | nan           |        13 |
|     42 | M3_circ         | test_ood_ecc    | T1       | S3b_diff |  -0.130589    |       1.36941     |   0.000385708 |         0 |
|     42 | M3_circ         | test_ood_ecc    | T1       | S3b_std  |   0.00350367  |       1.5035      |   9.98395e-07 |         0 |
|     42 | M3_circ         | test_ood_ecc    | T3       | S2       |  -0.756805    |       0.743195    |   0.238992    |         0 |
|     42 | M3_circ         | test_ood_ecc    | T3       | S3a_diff | nan           |     nan           | nan           |         1 |
|     42 | M3_circ         | test_ood_ecc    | T3       | S3b_diff |   1.64737     |       3.14737     |   0.0155443   |         0 |
|     42 | M3_circ         | test_ood_ecc    | T3       | S3b_std  |  -0.821708    |       0.678292    |   0.175098    |         0 |
|     42 | M3_circ         | test_iso_energy | T3       | S2       |  -1.09868     |       0.401321    |   0.313716    |         0 |
|     42 | M3_circ         | test_iso_energy | T3       | S3a_diff |  -1.23        |       0.269999    |   0.304935    |         0 |
|     42 | M3_circ         | test_iso_energy | T3       | S3b_diff |  -1.19686     |       0.303145    |   0.283099    |         0 |
|     42 | M3_circ         | test_iso_energy | T3       | S3b_std  |  -1.24997     |       0.250029    |   0.313934    |         0 |
|     42 | M3_circ         | test_ellip_peri | T3       | S2       |   0.220141    |       1.72014     |   0.0947903   |         0 |
|     42 | M3_circ         | test_ellip_peri | T3       | S3a_diff |   0.213618    |       1.71362     |   0.0912333   |         0 |
|     42 | M3_circ         | test_ellip_peri | T3       | S3b_diff |   0.217336    |       1.71734     |   0.0932511   |         0 |
|     42 | M3_circ         | test_ellip_peri | T3       | S3b_std  |   0.216833    |       1.71683     |   0.0925993   |         0 |
|     42 | M3_circ         | test_ellip_apo  | T3       | S2       |  -1.46331     |       0.0366926   |   0.565388    |         0 |
|     42 | M3_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.6957      |       0.195703    |   0.711965    |         0 |
|     42 | M3_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.58474     |       0.0847352   |   0.624974    |         0 |
|     42 | M3_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.57933     |       0.0793275   |   0.619285    |         0 |
|     42 | M1_Ellip_Short  | test_id_ellip   | T1       | S2       |  -1.44319     |       0.0568116   |   0.993194    |         0 |
|     42 | M1_Ellip_Short  | test_id_ellip   | T1       | S3a_diff |  -1.77731     |       0.277311    |   0.839586    |         0 |
|     42 | M1_Ellip_Short  | test_id_ellip   | T1       | S3b_diff |  -1.50166     |       0.00166471  |   0.999637    |         0 |
|     42 | M1_Ellip_Short  | test_id_ellip   | T1       | S3b_std  |  -1.50306     |       0.00305789  |   0.999746    |         0 |
|     42 | M1_Ellip_Short  | test_id_ellip   | T3       | S2       |  -1.2282      |       0.271802    |   0.76724     |         0 |
|     42 | M1_Ellip_Short  | test_id_ellip   | T3       | S3a_diff |  -1.44885     |       0.0511483   |   0.758579    |         0 |
|     42 | M1_Ellip_Short  | test_id_ellip   | T3       | S3b_diff |  -1.4548      |       0.0451985   |   0.757332    |         0 |
|     42 | M1_Ellip_Short  | test_id_ellip   | T3       | S3b_std  |  -1.45653     |       0.0434694   |   0.756799    |         0 |
|     42 | M1_Ellip_Short  | test_ood_ecc    | T1       | S2       |  -1.44412     |       0.0558777   |   0.992922    |         0 |
|     42 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3a_diff |  -1.83964     |       0.339641    |   0.855446    |         0 |
|     42 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3b_diff |  -2.10677     |       0.606765    |   0.94415     |         0 |
|     42 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3b_std  |  -1.50203     |       0.00203017  |   0.999774    |         0 |
|     42 | M1_Ellip_Short  | test_ood_ecc    | T3       | S2       |  -1.2643      |       0.235699    |   0.505832    |         0 |
|     42 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3a_diff |  -1.77526     |       0.275262    |   0.643189    |         0 |
|     42 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3b_diff |  -1.78296     |       0.282964    |   0.643162    |         0 |
|     42 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3b_std  |  -1.69172     |       0.191725    |   0.624512    |         0 |
|     42 | M1_Ellip_Short  | test_iso_energy | T3       | S2       |  -1.45201     |       0.0479945   |   0.731284    |         0 |
|     42 | M1_Ellip_Short  | test_iso_energy | T3       | S3a_diff |  -1.65274     |       0.152744    |   0.741715    |         0 |
|     42 | M1_Ellip_Short  | test_iso_energy | T3       | S3b_diff |  -1.65612     |       0.156119    |   0.741302    |         0 |
|     42 | M1_Ellip_Short  | test_iso_energy | T3       | S3b_std  |  -1.64271     |       0.142707    |   0.744238    |         0 |
|     42 | M1_Ellip_Short  | test_ellip_peri | T3       | S2       |  -0.708251    |       0.791749    |   0.939195    |         0 |
|     42 | M1_Ellip_Short  | test_ellip_peri | T3       | S3a_diff |  -0.754685    |       0.745315    |   0.986151    |         0 |
|     42 | M1_Ellip_Short  | test_ellip_peri | T3       | S3b_diff |  -0.729367    |       0.770633    |   0.969889    |         0 |
|     42 | M1_Ellip_Short  | test_ellip_peri | T3       | S3b_std  |  -0.730475    |       0.769525    |   0.970089    |         0 |
|     42 | M1_Ellip_Short  | test_ellip_apo  | T3       | S2       |  -1.45485     |       0.0451548   |   0.865827    |         0 |
|     42 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3a_diff |  -2.21447     |       0.714475    |   0.998332    |         0 |
|     42 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3b_diff |  -2.21487     |       0.71487     |   0.998275    |         0 |
|     42 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3b_std  |  -1.98066     |       0.480661    |   0.976402    |         0 |
|     42 | M1_Ellip_Inv    | test_id_ellip   | T1       | S2       |  -1.43881     |       0.0611875   |   0.993468    |         0 |
|     42 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3a_diff |  -1.65975     |       0.15975     |   0.855008    |         0 |
|     42 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3b_diff |  -1.66029     |       0.160285    |   0.954105    |         0 |
|     42 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3b_std  |  -1.50328     |       0.00327943  |   0.999728    |         0 |
|     42 | M1_Ellip_Inv    | test_id_ellip   | T3       | S2       |  -1.43501     |       0.0649947   |   0.987784    |         0 |
|     42 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3a_diff |  -1.50882     |       0.00882229  |   0.992378    |         0 |
|     42 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3b_diff |  -1.50981     |       0.00980571  |   0.992276    |         0 |
|     42 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3b_std  |  -1.51414     |       0.0141379   |   0.991774    |         0 |
|     42 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S2       |  -1.44381     |       0.0561919   |   0.99264     |         0 |
|     42 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3a_diff |  -2.31973     |       0.819735    |   0.77731     |         0 |
|     42 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3b_diff |  -2.12497     |       0.624975    |   0.93442     |         0 |
|     42 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3b_std  |  -1.57364     |       0.0736448   |   0.96502     |         0 |
|     42 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S2       |  -1.44285     |       0.0571509   |   0.979009    |         0 |
|     42 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3a_diff |  -1.56392     |       0.063917    |   0.984977    |         0 |
|     42 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3b_diff |  -1.56904     |       0.0690392   |   0.984707    |         0 |
|     42 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3b_std  |  -1.58569     |       0.0856858   |   0.984077    |         0 |
|     42 | M1_Ellip_Inv    | test_iso_energy | T3       | S2       |  -1.40336     |       0.0966443   |   0.992492    |         0 |
|     42 | M1_Ellip_Inv    | test_iso_energy | T3       | S3a_diff |  -1.55877     |       0.0587736   |   0.987558    |         0 |
|     42 | M1_Ellip_Inv    | test_iso_energy | T3       | S3b_diff |  -1.56104     |       0.0610438   |   0.987188    |         0 |
|     42 | M1_Ellip_Inv    | test_iso_energy | T3       | S3b_std  |  -1.57705     |       0.0770482   |   0.984444    |         0 |
|     42 | M1_Ellip_Inv    | test_ellip_peri | T3       | S2       |  -1.43482     |       0.0651793   |   0.980757    |         0 |
|     42 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3a_diff |  -1.5243      |       0.0242967   |   0.995215    |         0 |
|     42 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3b_diff |  -1.5247      |       0.0247024   |   0.995217    |         0 |
|     42 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3b_std  |  -1.53087     |       0.0308655   |   0.995575    |         0 |
|     42 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S2       |  -1.4424      |       0.0576039   |   0.99388     |         0 |
|     42 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3a_diff |  -1.54161     |       0.0416125   |   0.997437    |         0 |
|     42 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3b_diff |  -1.54454     |       0.0445386   |   0.997172    |         0 |
|     42 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3b_std  |  -1.55387     |       0.05387     |   0.99584     |         0 |
|     42 | M2_Ellip_Inv    | test_id_ellip   | T1       | S2       |  -1.4395      |       0.0604999   |   0.993389    |         0 |
|     42 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3a_diff |  -1.59876     |       0.098755    |   0.876675    |         0 |
|     42 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3b_diff |  -1.5011      |       0.00109523  |   0.999501    |         0 |
|     42 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3b_std  |  -1.50321     |       0.00321067  |   0.999725    |         0 |
|     42 | M2_Ellip_Inv    | test_id_ellip   | T3       | S2       |  -1.43634     |       0.0636604   |   0.990162    |         0 |
|     42 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3a_diff |  -1.51791     |       0.0179149   |   0.995473    |         0 |
|     42 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3b_diff |  -1.52117     |       0.0211737   |   0.995282    |         0 |
|     42 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3b_std  |  -1.53203     |       0.0320312   |   0.994299    |         0 |
|     42 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S2       |  -1.44485     |       0.0551512   |   0.992605    |         0 |
|     42 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3a_diff |  -2.43092     |       0.930915    |   0.844588    |         0 |
|     42 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3b_diff |  -2.11023     |       0.610235    |   0.9374      |         0 |
|     42 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3b_std  |  -1.56088     |       0.0608764   |   0.989351    |         0 |
|     42 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S2       |  -1.51377     |       0.013774    |   0.958284    |         0 |
|     42 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3a_diff | nan           |     nan           | nan           |         2 |
|     42 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3b_diff |  -1.10821     |       0.39179     |   0.00372544  |         0 |
|     42 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3b_std  |  -1.44903     |       0.0509678   |   0.615812    |         0 |
|     42 | M2_Ellip_Inv    | test_iso_energy | T3       | S2       |  -1.41318     |       0.0868157   |   0.993623    |         0 |
|     42 | M2_Ellip_Inv    | test_iso_energy | T3       | S3a_diff |  -1.55926     |       0.0592578   |   0.989229    |         0 |
|     42 | M2_Ellip_Inv    | test_iso_energy | T3       | S3b_diff |  -1.56359     |       0.0635918   |   0.988708    |         0 |
|     42 | M2_Ellip_Inv    | test_iso_energy | T3       | S3b_std  |  -1.58951     |       0.0895057   |   0.984594    |         0 |
|     42 | M2_Ellip_Inv    | test_ellip_peri | T3       | S2       |  -1.41318     |       0.0868215   |   0.990121    |         0 |
|     42 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3a_diff |  -1.46614     |       0.0338615   |   0.999403    |         0 |
|     42 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3b_diff |  -1.46678     |       0.0332199   |   0.999412    |         0 |
|     42 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3b_std  |  -1.47004     |       0.0299608   |   0.999576    |         0 |
|     42 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S2       |  -1.47266     |       0.0273403   |   0.99034     |         0 |
|     42 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3a_diff |  -1.58528     |       0.0852814   |   0.998732    |         0 |
|     42 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3b_diff |  -1.59406     |       0.0940578   |   0.998318    |         0 |
|     42 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3b_std  |  -1.61888     |       0.118879    |   0.996233    |         0 |
|     42 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S2       |  -1.42351     |       0.076492    |   0.79764     |         0 |
|     42 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3a_diff | nan           |     nan           | nan           |         4 |
|     42 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3b_diff |  -1.86567     |       0.365674    |   0.206659    |         0 |
|     42 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3b_std  |  -1.45385     |       0.046147    |   0.148099    |         0 |
|     42 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S2       |   0.000155511 |       1.50016     |   0.0288138   |         0 |
|     42 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3a_diff | nan           |     nan           | nan           |         4 |
|     42 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3b_diff |   2.15421     |       3.65421     |   0.0175146   |         0 |
|     42 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3b_std  |   0.0203339   |       1.52033     |   0.0805937   |         0 |
|     42 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S2       |  -1.44854     |       0.0514575   |   0.992707    |         0 |
|     42 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3a_diff | nan           |     nan           | nan           |         8 |
|     42 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3b_diff |  -0.991671    |       0.508329    |   0.0784866   |         0 |
|     42 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3b_std  |  -1.93461     |       0.434611    |   0.906555    |         0 |
|     42 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S2       |   0.000313224 |       1.50031     |   0.119904    |         0 |
|     42 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3a_diff | nan           |     nan           | nan           |         3 |
|     42 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3b_diff |  -5.87695     |       4.37695     |   0.0974854   |         0 |
|     42 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3b_std  |  -0.00368951  |       1.49631     |   0.00296108  |         0 |
|     42 | M3_Ellip_SIGReg | test_iso_energy | T3       | S2       |   0.0003116   |       1.50031     |   0.101788    |         0 |
|     42 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3a_diff | nan           |     nan           | nan           |         7 |
|     42 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3b_diff |  -6.12037     |       4.62037     |   0.0748766   |         0 |
|     42 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3b_std  |  -0.00440307  |       1.4956      |   0.00273991  |         0 |
|     42 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S2       |   8.07514e-05 |       1.50008     |   0.0664641   |         0 |
|     42 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3a_diff |   0.0969543   |       1.59695     |   0.132304    |         0 |
|     42 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3b_diff |   0.0832206   |       1.58322     |   0.193754    |         0 |
|     42 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3b_std  |   0.00743844  |       1.50744     |   0.148144    |         0 |
|     42 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S2       |   0.000774403 |       1.50077     |   0.429797    |         0 |
|     42 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3a_diff | nan           |     nan           | nan           |        10 |
|     42 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3b_diff | -16.6417      |      15.1417      |   0.433301    |         0 |
|     42 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3b_std  |  -0.0245467   |       1.47545     |   0.185095    |         0 |
|     43 | M0_Koopman      | test_id_ellip   | T1       | S2       |  -1.4559      |       0.0441019   |   0.991465    |         0 |
|     43 | M0_Koopman      | test_id_ellip   | T1       | S3a_diff |  -1.77822     |       0.278222    |   0.816047    |         0 |
|     43 | M0_Koopman      | test_id_ellip   | T1       | S3b_diff |  -1.49566     |       0.00434278  |   0.999424    |         0 |
|     43 | M0_Koopman      | test_id_ellip   | T1       | S3b_std  |  -1.49962     |       0.000375594 |   0.999601    |         0 |
|     43 | M0_Koopman      | test_id_ellip   | T3       | S2       |   0.00713211  |       1.50713     |   0.130844    |         0 |
|     43 | M0_Koopman      | test_id_ellip   | T3       | S3a_diff |   0.00447955  |       1.50448     |   0.067303    |         0 |
|     43 | M0_Koopman      | test_id_ellip   | T3       | S3b_diff |   0.00462074  |       1.50462     |   0.075624    |         0 |
|     43 | M0_Koopman      | test_id_ellip   | T3       | S3b_std  |   0.00656461  |       1.50656     |   0.112574    |         0 |
|     43 | M0_Koopman      | test_ood_ecc    | T1       | S2       |  -1.42023     |       0.0797677   |   0.994158    |         0 |
|     43 | M0_Koopman      | test_ood_ecc    | T1       | S3a_diff |  -1.52999     |       0.0299942   |   0.874296    |         0 |
|     43 | M0_Koopman      | test_ood_ecc    | T1       | S3b_diff |  -1.51034     |       0.0103379   |   0.997899    |         0 |
|     43 | M0_Koopman      | test_ood_ecc    | T1       | S3b_std  |  -1.50373     |       0.00373445  |   0.999711    |         0 |
|     43 | M0_Koopman      | test_ood_ecc    | T3       | S2       |   0.00358428  |       1.50358     |   0.0233322   |         0 |
|     43 | M0_Koopman      | test_ood_ecc    | T3       | S3a_diff |   0.00289227  |       1.50289     |   0.0124779   |         0 |
|     43 | M0_Koopman      | test_ood_ecc    | T3       | S3b_diff |   0.00275855  |       1.50276     |   0.0120375   |         0 |
|     43 | M0_Koopman      | test_ood_ecc    | T3       | S3b_std  |   0.00365205  |       1.50365     |   0.0209135   |         0 |
|     43 | M0_Koopman      | test_iso_energy | T3       | S2       |   0.0080533   |       1.50805     |   0.121237    |         0 |
|     43 | M0_Koopman      | test_iso_energy | T3       | S3a_diff |   0.00694909  |       1.50695     |   0.0794565   |         0 |
|     43 | M0_Koopman      | test_iso_energy | T3       | S3b_diff |   0.00684388  |       1.50684     |   0.0824002   |         0 |
|     43 | M0_Koopman      | test_iso_energy | T3       | S3b_std  |   0.00801218  |       1.50801     |   0.108734    |         0 |
|     43 | M0_Koopman      | test_ellip_peri | T3       | S2       |  -0.0063603   |       1.49364     |   0.165698    |         0 |
|     43 | M0_Koopman      | test_ellip_peri | T3       | S3a_diff |  -0.00199628  |       1.498       |   0.00862826  |         0 |
|     43 | M0_Koopman      | test_ellip_peri | T3       | S3b_diff |  -0.0023365   |       1.49766     |   0.0132745   |         0 |
|     43 | M0_Koopman      | test_ellip_peri | T3       | S3b_std  |  -0.00497564  |       1.49502     |   0.0733047   |         0 |
|     43 | M0_Koopman      | test_ellip_apo  | T3       | S2       |   0.0214522   |       1.52145     |   0.969631    |         0 |
|     43 | M0_Koopman      | test_ellip_apo  | T3       | S3a_diff |   0.0218891   |       1.52189     |   0.891541    |         0 |
|     43 | M0_Koopman      | test_ellip_apo  | T3       | S3b_diff |   0.0216336   |       1.52163     |   0.895332    |         0 |
|     43 | M0_Koopman      | test_ellip_apo  | T3       | S3b_std  |   0.022504    |       1.5225      |   0.947314    |         0 |
|     43 | M1_circ         | test_id_ellip   | T1       | S2       |  -1.456       |       0.0440031   |   0.991455    |         0 |
|     43 | M1_circ         | test_id_ellip   | T1       | S3a_diff |  -1.77357     |       0.273569    |   0.817476    |         0 |
|     43 | M1_circ         | test_id_ellip   | T1       | S3b_diff |  -1.49592     |       0.00408091  |   0.999418    |         0 |
|     43 | M1_circ         | test_id_ellip   | T1       | S3b_std  |  -1.49958     |       0.000423034 |   0.9996      |         0 |
|     43 | M1_circ         | test_id_ellip   | T3       | S2       |  -1.27876     |       0.221237    |   0.732246    |         0 |
|     43 | M1_circ         | test_id_ellip   | T3       | S3a_diff |  -1.49047     |       0.00953056  |   0.705643    |         0 |
|     43 | M1_circ         | test_id_ellip   | T3       | S3b_diff |  -1.49057     |       0.00942626  |   0.705576    |         0 |
|     43 | M1_circ         | test_id_ellip   | T3       | S3b_std  |  -1.47593     |       0.0240739   |   0.712547    |         0 |
|     43 | M1_circ         | test_ood_ecc    | T1       | S2       |  -1.42052     |       0.0794845   |   0.994154    |         0 |
|     43 | M1_circ         | test_ood_ecc    | T1       | S3a_diff |  -1.52752     |       0.0275186   |   0.87398     |         0 |
|     43 | M1_circ         | test_ood_ecc    | T1       | S3b_diff |  -1.51088     |       0.0108759   |   0.997929    |         0 |
|     43 | M1_circ         | test_ood_ecc    | T1       | S3b_std  |  -1.50396     |       0.00395546  |   0.999715    |         0 |
|     43 | M1_circ         | test_ood_ecc    | T3       | S2       |  -0.239047    |       1.26095     |   0.0263927   |         0 |
|     43 | M1_circ         | test_ood_ecc    | T3       | S3a_diff |  -0.663149    |       0.836851    |   0.117092    |         0 |
|     43 | M1_circ         | test_ood_ecc    | T3       | S3b_diff |  -0.662887    |       0.837113    |   0.116961    |         0 |
|     43 | M1_circ         | test_ood_ecc    | T3       | S3b_std  |  -0.499077    |       1.00092     |   0.0826237   |         0 |
|     43 | M1_circ         | test_iso_energy | T3       | S2       |  -1.13239     |       0.367614    |   0.488076    |         0 |
|     43 | M1_circ         | test_iso_energy | T3       | S3a_diff |  -1.49136     |       0.00863788  |   0.528805    |         0 |
|     43 | M1_circ         | test_iso_energy | T3       | S3b_diff |  -1.49191     |       0.00808628  |   0.528793    |         0 |
|     43 | M1_circ         | test_iso_energy | T3       | S3b_std  |  -1.41466     |       0.0853359   |   0.527245    |         0 |
|     43 | M1_circ         | test_ellip_peri | T3       | S2       |  -0.346379    |       1.15362     |   0.248365    |         0 |
|     43 | M1_circ         | test_ellip_peri | T3       | S3a_diff |  -0.347023    |       1.15298     |   0.249664    |         0 |
|     43 | M1_circ         | test_ellip_peri | T3       | S3b_diff |  -0.346876    |       1.15312     |   0.249464    |         0 |
|     43 | M1_circ         | test_ellip_peri | T3       | S3b_std  |  -0.345639    |       1.15436     |   0.247829    |         0 |
|     43 | M1_circ         | test_ellip_apo  | T3       | S2       |  -0.854043    |       0.645957    |   0.836686    |         0 |
|     43 | M1_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.46015     |       0.0398512   |   0.999921    |         0 |
|     43 | M1_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.4612      |       0.0387967   |   0.999925    |         0 |
|     43 | M1_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.3458      |       0.154199    |   0.987582    |         0 |
|     43 | M2_circ         | test_id_ellip   | T1       | S2       |  -1.45562     |       0.0443774   |   0.991511    |         0 |
|     43 | M2_circ         | test_id_ellip   | T1       | S3a_diff |  -1.76996     |       0.269958    |   0.828401    |         0 |
|     43 | M2_circ         | test_id_ellip   | T1       | S3b_diff |  -1.49637     |       0.00362781  |   0.999397    |         0 |
|     43 | M2_circ         | test_id_ellip   | T1       | S3b_std  |  -1.49994     |       6.23371e-05 |   0.999593    |         0 |
|     43 | M2_circ         | test_id_ellip   | T3       | S2       |  -1.13097     |       0.36903     |   0.721775    |         0 |
|     43 | M2_circ         | test_id_ellip   | T3       | S3a_diff |  -1.33375     |       0.166246    |   0.676659    |         0 |
|     43 | M2_circ         | test_id_ellip   | T3       | S3b_diff |  -1.33961     |       0.160388    |   0.674995    |         0 |
|     43 | M2_circ         | test_id_ellip   | T3       | S3b_std  |  -1.33031     |       0.169692    |   0.684122    |         0 |
|     43 | M2_circ         | test_ood_ecc    | T1       | S2       |  -1.42172     |       0.0782847   |   0.994268    |         0 |
|     43 | M2_circ         | test_ood_ecc    | T1       | S3a_diff |  -1.6285      |       0.128497    |   0.875454    |         0 |
|     43 | M2_circ         | test_ood_ecc    | T1       | S3b_diff |  -1.51177     |       0.011772    |   0.998033    |         0 |
|     43 | M2_circ         | test_ood_ecc    | T1       | S3b_std  |  -1.50425     |       0.00425229  |   0.99974     |         0 |
|     43 | M2_circ         | test_ood_ecc    | T3       | S2       |  -0.201434    |       1.29857     |   0.0236068   |         0 |
|     43 | M2_circ         | test_ood_ecc    | T3       | S3a_diff |  -0.640709    |       0.859291    |   0.126603    |         0 |
|     43 | M2_circ         | test_ood_ecc    | T3       | S3b_diff |  -0.639719    |       0.860281    |   0.126107    |         0 |
|     43 | M2_circ         | test_ood_ecc    | T3       | S3b_std  |  -0.469652    |       1.03035     |   0.0872091   |         0 |
|     43 | M2_circ         | test_iso_energy | T3       | S2       |  -1.01579     |       0.484215    |   0.439992    |         0 |
|     43 | M2_circ         | test_iso_energy | T3       | S3a_diff |  -1.36208     |       0.13792     |   0.480004    |         0 |
|     43 | M2_circ         | test_iso_energy | T3       | S3b_diff |  -1.36878     |       0.131223    |   0.480409    |         0 |
|     43 | M2_circ         | test_iso_energy | T3       | S3b_std  |  -1.30354     |       0.196464    |   0.485711    |         0 |
|     43 | M2_circ         | test_ellip_peri | T3       | S2       |  -0.124343    |       1.37566     |   0.0379537   |         0 |
|     43 | M2_circ         | test_ellip_peri | T3       | S3a_diff |  -0.125822    |       1.37418     |   0.0390736   |         0 |
|     43 | M2_circ         | test_ellip_peri | T3       | S3b_diff |  -0.125762    |       1.37424     |   0.0390315   |         0 |
|     43 | M2_circ         | test_ellip_peri | T3       | S3b_std  |  -0.123886    |       1.37611     |   0.0378356   |         0 |
|     43 | M2_circ         | test_ellip_apo  | T3       | S2       |  -0.814862    |       0.685138    |   0.849991    |         0 |
|     43 | M2_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.38478     |       0.115218    |   0.999351    |         0 |
|     43 | M2_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.38888     |       0.111118    |   0.999368    |         0 |
|     43 | M2_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.28948     |       0.210518    |   0.991982    |         0 |
|     43 | M3_circ         | test_id_ellip   | T1       | S2       |  -1.23296     |       0.267037    |   0.744484    |         0 |
|     43 | M3_circ         | test_id_ellip   | T1       | S3a_diff | nan           |     nan           | nan           |         2 |
|     43 | M3_circ         | test_id_ellip   | T1       | S3b_diff |   1.76523     |       3.26523     |   0.0912818   |         0 |
|     43 | M3_circ         | test_id_ellip   | T1       | S3b_std  |  -1.44578     |       0.0542208   |   0.0702658   |         0 |
|     43 | M3_circ         | test_id_ellip   | T3       | S2       |  -0.649242    |       0.850758    |   0.147728    |         0 |
|     43 | M3_circ         | test_id_ellip   | T3       | S3a_diff | nan           |     nan           | nan           |         1 |
|     43 | M3_circ         | test_id_ellip   | T3       | S3b_diff |  -0.933975    |       0.566025    |   0.15375     |         0 |
|     43 | M3_circ         | test_id_ellip   | T3       | S3b_std  |  -0.785391    |       0.714609    |   0.149683    |         0 |
|     43 | M3_circ         | test_ood_ecc    | T1       | S2       |  -1.45627     |       0.0437255   |   0.99151     |         0 |
|     43 | M3_circ         | test_ood_ecc    | T1       | S3a_diff | nan           |     nan           | nan           |         4 |
|     43 | M3_circ         | test_ood_ecc    | T1       | S3b_diff |   3.09772     |       4.59772     |   0.276509    |         0 |
|     43 | M3_circ         | test_ood_ecc    | T1       | S3b_std  |  -0.080483    |       1.41952     |   0.000509274 |         0 |
|     43 | M3_circ         | test_ood_ecc    | T3       | S2       |  -0.345946    |       1.15405     |   0.0524938   |         0 |
|     43 | M3_circ         | test_ood_ecc    | T3       | S3a_diff |  -0.289607    |       1.21039     |   0.0419341   |         0 |
|     43 | M3_circ         | test_ood_ecc    | T3       | S3b_diff |  -0.217073    |       1.28293     |   0.0212767   |         0 |
|     43 | M3_circ         | test_ood_ecc    | T3       | S3b_std  |  -0.488337    |       1.01166     |   0.0845311   |         0 |
|     43 | M3_circ         | test_iso_energy | T3       | S2       |  -0.800689    |       0.699311    |   0.153096    |         0 |
|     43 | M3_circ         | test_iso_energy | T3       | S3a_diff | nan           |     nan           | nan           |         2 |
|     43 | M3_circ         | test_iso_energy | T3       | S3b_diff |  -4.627       |       3.127       |   0.0959853   |         0 |
|     43 | M3_circ         | test_iso_energy | T3       | S3b_std  |  -1.10504     |       0.394957    |   0.186363    |         0 |
|     43 | M3_circ         | test_ellip_peri | T3       | S2       |   0.524556    |       2.02456     |   0.22768     |         0 |
|     43 | M3_circ         | test_ellip_peri | T3       | S3a_diff |   0.546178    |       2.04618     |   0.260281    |         0 |
|     43 | M3_circ         | test_ellip_peri | T3       | S3b_diff |   0.519582    |       2.01958     |   0.22379     |         0 |
|     43 | M3_circ         | test_ellip_peri | T3       | S3b_std  |   0.520076    |       2.02008     |   0.223307    |         0 |
|     43 | M3_circ         | test_ellip_apo  | T3       | S2       |  -1.24492     |       0.255084    |   0.650001    |         0 |
|     43 | M3_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.59176     |       0.0917559   |   0.763098    |         0 |
|     43 | M3_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.59038     |       0.0903758   |   0.684412    |         0 |
|     43 | M3_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.59188     |       0.0918846   |   0.672423    |         0 |
|     43 | M1_Ellip_Short  | test_id_ellip   | T1       | S2       |  -1.45408     |       0.0459212   |   0.991927    |         0 |
|     43 | M1_Ellip_Short  | test_id_ellip   | T1       | S3a_diff |  -1.77914     |       0.279143    |   0.842032    |         0 |
|     43 | M1_Ellip_Short  | test_id_ellip   | T1       | S3b_diff |  -1.49636     |       0.00363624  |   0.999716    |         0 |
|     43 | M1_Ellip_Short  | test_id_ellip   | T1       | S3b_std  |  -1.50226     |       0.00226324  |   0.999754    |         0 |
|     43 | M1_Ellip_Short  | test_id_ellip   | T3       | S2       |  -1.22019     |       0.279812    |   0.730447    |         0 |
|     43 | M1_Ellip_Short  | test_id_ellip   | T3       | S3a_diff |  -1.36015     |       0.139854    |   0.727164    |         0 |
|     43 | M1_Ellip_Short  | test_id_ellip   | T3       | S3b_diff |  -1.35999     |       0.140008    |   0.726533    |         0 |
|     43 | M1_Ellip_Short  | test_id_ellip   | T3       | S3b_std  |  -1.36925     |       0.130751    |   0.729732    |         0 |
|     43 | M1_Ellip_Short  | test_ood_ecc    | T1       | S2       |  -1.43196     |       0.068041    |   0.994322    |         0 |
|     43 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3a_diff |  -2.06424     |       0.564238    |   0.84792     |         0 |
|     43 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3b_diff |  -1.99269     |       0.492685    |   0.919873    |         0 |
|     43 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3b_std  |  -1.50459     |       0.0045946   |   0.999802    |         0 |
|     43 | M1_Ellip_Short  | test_ood_ecc    | T3       | S2       |  -1.00552     |       0.494479    |   0.361754    |         0 |
|     43 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3a_diff |  -1.38732     |       0.112681    |   0.436943    |         0 |
|     43 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3b_diff |  -1.40485     |       0.0951452   |   0.443476    |         0 |
|     43 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3b_std  |  -1.29521     |       0.20479     |   0.429085    |         0 |
|     43 | M1_Ellip_Short  | test_iso_energy | T3       | S2       |  -1.37745     |       0.122554    |   0.751844    |         0 |
|     43 | M1_Ellip_Short  | test_iso_energy | T3       | S3a_diff |  -1.68714     |       0.187138    |   0.755823    |         0 |
|     43 | M1_Ellip_Short  | test_iso_energy | T3       | S3b_diff |  -1.69106     |       0.191063    |   0.755517    |         0 |
|     43 | M1_Ellip_Short  | test_iso_energy | T3       | S3b_std  |  -1.64264     |       0.142642    |   0.771066    |         0 |
|     43 | M1_Ellip_Short  | test_ellip_peri | T3       | S2       |  -0.791406    |       0.708594    |   0.997218    |         0 |
|     43 | M1_Ellip_Short  | test_ellip_peri | T3       | S3a_diff |  -0.744132    |       0.755868    |   0.998273    |         0 |
|     43 | M1_Ellip_Short  | test_ellip_peri | T3       | S3b_diff |  -0.766996    |       0.733004    |   0.998796    |         0 |
|     43 | M1_Ellip_Short  | test_ellip_peri | T3       | S3b_std  |  -0.77414     |       0.72586     |   0.998919    |         0 |
|     43 | M1_Ellip_Short  | test_ellip_apo  | T3       | S2       |  -1.57288     |       0.0728832   |   0.888222    |         0 |
|     43 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3a_diff |  -2.15273     |       0.652726    |   0.995562    |         0 |
|     43 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3b_diff |  -2.16376     |       0.663759    |   0.995239    |         0 |
|     43 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3b_std  |  -2.05418     |       0.55418     |   0.976941    |         0 |
|     43 | M1_Ellip_Inv    | test_id_ellip   | T1       | S2       |  -1.45502     |       0.0449817   |   0.991565    |         0 |
|     43 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3a_diff |  -1.63933     |       0.139332    |   0.819411    |         0 |
|     43 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3b_diff |  -1.57604     |       0.0760411   |   0.965466    |         0 |
|     43 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3b_std  |  -1.50178     |       0.00178146  |   0.999659    |         0 |
|     43 | M1_Ellip_Inv    | test_id_ellip   | T3       | S2       |  -1.42776     |       0.0722364   |   0.989161    |         0 |
|     43 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3a_diff |  -1.48998     |       0.0100207   |   0.994273    |         0 |
|     43 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3b_diff |  -1.4903      |       0.00970032  |   0.994263    |         0 |
|     43 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3b_std  |  -1.49464     |       0.00536091  |   0.993774    |         0 |
|     43 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S2       |  -1.42299     |       0.0770057   |   0.994193    |         0 |
|     43 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3a_diff |  -2.38852     |       0.888519    |   0.863425    |         0 |
|     43 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3b_diff |  -2.17108     |       0.67108     |   0.941315    |         0 |
|     43 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3b_std  |  -1.70535     |       0.205355    |   0.949216    |         0 |
|     43 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S2       |  -1.33892     |       0.161076    |   0.944059    |         0 |
|     43 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3a_diff |  -1.35714     |       0.142861    |   0.943541    |         0 |
|     43 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3b_diff |  -1.35794     |       0.142063    |   0.943482    |         0 |
|     43 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3b_std  |  -1.35735     |       0.142653    |   0.942891    |         0 |
|     43 | M1_Ellip_Inv    | test_iso_energy | T3       | S2       |  -1.40981     |       0.0901859   |   0.990657    |         0 |
|     43 | M1_Ellip_Inv    | test_iso_energy | T3       | S3a_diff |  -1.52921     |       0.0292057   |   0.989442    |         0 |
|     43 | M1_Ellip_Inv    | test_iso_energy | T3       | S3b_diff |  -1.52994     |       0.0299432   |   0.989396    |         0 |
|     43 | M1_Ellip_Inv    | test_iso_energy | T3       | S3b_std  |  -1.54086     |       0.0408595   |   0.988163    |         0 |
|     43 | M1_Ellip_Inv    | test_ellip_peri | T3       | S2       |  -1.35878     |       0.141218    |   0.989371    |         0 |
|     43 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3a_diff |  -1.3822      |       0.117797    |   0.994426    |         0 |
|     43 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3b_diff |  -1.38235     |       0.117649    |   0.994427    |         0 |
|     43 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3b_std  |  -1.38119     |       0.11881     |   0.994436    |         0 |
|     43 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S2       |  -1.45571     |       0.0442906   |   0.991007    |         0 |
|     43 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3a_diff |  -1.54574     |       0.0457434   |   0.999452    |         0 |
|     43 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3b_diff |  -1.54692     |       0.046916    |   0.999414    |         0 |
|     43 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3b_std  |  -1.55407     |       0.0540659   |   0.998913    |         0 |
|     43 | M2_Ellip_Inv    | test_id_ellip   | T1       | S2       |  -1.45363     |       0.046373    |   0.991857    |         0 |
|     43 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3a_diff |  -1.60648     |       0.106479    |   0.842104    |         0 |
|     43 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3b_diff |  -1.57874     |       0.0787416   |   0.965731    |         0 |
|     43 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3b_std  |  -1.50372     |       0.00372397  |   0.999687    |         0 |
|     43 | M2_Ellip_Inv    | test_id_ellip   | T3       | S2       |  -1.40961     |       0.0903935   |   0.990128    |         0 |
|     43 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3a_diff |  -1.47666     |       0.0233374   |   0.995151    |         0 |
|     43 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3b_diff |  -1.47752     |       0.0224848   |   0.995116    |         0 |
|     43 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3b_std  |  -1.48336     |       0.0166429   |   0.994542    |         0 |
|     43 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S2       |  -1.42381     |       0.0761892   |   0.994279    |         0 |
|     43 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3a_diff |  -2.50943     |       1.00943     |   0.829418    |         0 |
|     43 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3b_diff |  -2.14719     |       0.647189    |   0.928834    |         0 |
|     43 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3b_std  |  -1.64046     |       0.140462    |   0.95897     |         0 |
|     43 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S2       |  -1.29738     |       0.202618    |   0.945281    |         0 |
|     43 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3a_diff |  -1.31612     |       0.183884    |   0.948001    |         0 |
|     43 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3b_diff |  -1.31374     |       0.186262    |   0.948014    |         0 |
|     43 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3b_std  |  -1.32096     |       0.179039    |   0.945008    |         0 |
|     43 | M2_Ellip_Inv    | test_iso_energy | T3       | S2       |  -1.39589     |       0.104108    |   0.996429    |         0 |
|     43 | M2_Ellip_Inv    | test_iso_energy | T3       | S3a_diff |  -1.50535     |       0.00534767  |   0.994776    |         0 |
|     43 | M2_Ellip_Inv    | test_iso_energy | T3       | S3b_diff |  -1.50862     |       0.00862392  |   0.994817    |         0 |
|     43 | M2_Ellip_Inv    | test_iso_energy | T3       | S3b_std  |  -1.52468     |       0.0246783   |   0.993383    |         0 |
|     43 | M2_Ellip_Inv    | test_ellip_peri | T3       | S2       |  -1.39232     |       0.107676    |   0.991883    |         0 |
|     43 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3a_diff |  -1.43119     |       0.0688064   |   0.998206    |         0 |
|     43 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3b_diff |  -1.43153     |       0.0684738   |   0.99821     |         0 |
|     43 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3b_std  |  -1.43299     |       0.0670095   |   0.998126    |         0 |
|     43 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S2       |  -1.4549      |       0.0450963   |   0.992047    |         0 |
|     43 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3a_diff |  -1.52239     |       0.0223859   |   0.999879    |         0 |
|     43 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3b_diff |  -1.52534     |       0.0253407   |   0.999858    |         0 |
|     43 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3b_std  |  -1.53567     |       0.0356707   |   0.999368    |         0 |
|     43 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S2       |  -1.50761     |       0.00760842  |   0.692667    |         0 |
|     43 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3a_diff | nan           |     nan           | nan           |         4 |
|     43 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3b_diff |  -0.613469    |       0.886531    |   0.0346329   |         0 |
|     43 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3b_std  |  -1.8256      |       0.325605    |   0.159704    |         0 |
|     43 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S2       |   6.88914e-05 |       1.50007     |   0.0359016   |         0 |
|     43 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3a_diff |   0.0821495   |       1.58215     |   0.035998    |         0 |
|     43 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3b_diff |   0.071217    |       1.57122     |   0.0368363   |         0 |
|     43 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3b_std  |   0.0126075   |       1.51261     |   0.0369022   |         0 |
|     43 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S2       |  -1.45223     |       0.0477733   |   0.992061    |         0 |
|     43 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3a_diff | nan           |     nan           | nan           |         4 |
|     43 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3b_diff |  -2.21319     |       0.71319     |   0.359389    |         0 |
|     43 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3b_std  |  -1.16708     |       0.332916    |   0.146853    |         0 |
|     43 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S2       |   7.6515e-05  |       1.50008     |   0.0461541   |         0 |
|     43 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3a_diff |   0.0682172   |       1.56822     |   0.0282161   |         0 |
|     43 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3b_diff |   0.0619256   |       1.56193     |   0.0305367   |         0 |
|     43 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3b_std  |   0.0113255   |       1.51133     |   0.0335296   |         0 |
|     43 | M3_Ellip_SIGReg | test_iso_energy | T3       | S2       |  -3.62129e-05 |       1.49996     |   0.00833297  |         0 |
|     43 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3a_diff |  -0.0392371   |       1.46076     |   0.00824822  |         0 |
|     43 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3b_diff |  -0.034814    |       1.46519     |   0.00853992  |         0 |
|     43 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3b_std  |  -0.0079107   |       1.49209     |   0.0122226   |         0 |
|     43 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S2       |   0.000204045 |       1.5002      |   0.516992    |         0 |
|     43 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3a_diff |   0.151514    |       1.65151     |   0.405165    |         0 |
|     43 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3b_diff |   0.143912    |       1.64391     |   0.428153    |         0 |
|     43 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3b_std  |   0.0257277   |       1.52573     |   0.485836    |         0 |
|     43 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S2       |  -0.000114909 |       1.49989     |   0.274773    |         0 |
|     43 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3a_diff |  -0.220912    |       1.27909     |   0.607023    |         0 |
|     43 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3b_diff |  -0.18108     |       1.31892     |   0.560121    |         0 |
|     43 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3b_std  |  -0.0264869   |       1.47351     |   0.498558    |         0 |
|     44 | M0_Koopman      | test_id_ellip   | T1       | S2       |  -1.43326     |       0.0667395   |   0.993772    |         0 |
|     44 | M0_Koopman      | test_id_ellip   | T1       | S3a_diff |  -1.79815     |       0.298154    |   0.832842    |         0 |
|     44 | M0_Koopman      | test_id_ellip   | T1       | S3b_diff |  -1.50903     |       0.00903274  |   0.99933     |         0 |
|     44 | M0_Koopman      | test_id_ellip   | T1       | S3b_std  |  -1.50647     |       0.00647367  |   0.999813    |         0 |
|     44 | M0_Koopman      | test_id_ellip   | T3       | S2       |   0.00743574  |       1.50744     |   0.0824107   |         0 |
|     44 | M0_Koopman      | test_id_ellip   | T3       | S3a_diff |   0.00469808  |       1.5047      |   0.0305371   |         0 |
|     44 | M0_Koopman      | test_id_ellip   | T3       | S3b_diff |   0.00453337  |       1.50453     |   0.0293798   |         0 |
|     44 | M0_Koopman      | test_id_ellip   | T3       | S3b_std  |   0.00706448  |       1.50706     |   0.0706004   |         0 |
|     44 | M0_Koopman      | test_ood_ecc    | T1       | S2       |  -1.42819     |       0.0718117   |   0.994264    |         0 |
|     44 | M0_Koopman      | test_ood_ecc    | T1       | S3a_diff |  -1.47606     |       0.0239436   |   0.879706    |         0 |
|     44 | M0_Koopman      | test_ood_ecc    | T1       | S3b_diff |  -1.49071     |       0.00928544  |   0.998723    |         0 |
|     44 | M0_Koopman      | test_ood_ecc    | T1       | S3b_std  |  -1.49612     |       0.00388229  |   0.999684    |         0 |
|     44 | M0_Koopman      | test_ood_ecc    | T3       | S2       |   0.0123004   |       1.5123      |   0.165888    |         0 |
|     44 | M0_Koopman      | test_ood_ecc    | T3       | S3a_diff |   0.00651071  |       1.50651     |   0.0475964   |         0 |
|     44 | M0_Koopman      | test_ood_ecc    | T3       | S3b_diff |   0.00667276  |       1.50667     |   0.0513254   |         0 |
|     44 | M0_Koopman      | test_ood_ecc    | T3       | S3b_std  |   0.010989    |       1.51099     |   0.131267    |         0 |
|     44 | M0_Koopman      | test_iso_energy | T3       | S2       |  -0.00260387  |       1.4974      |   0.00744945  |         0 |
|     44 | M0_Koopman      | test_iso_energy | T3       | S3a_diff |  -0.00575298  |       1.49425     |   0.0308353   |         0 |
|     44 | M0_Koopman      | test_iso_energy | T3       | S3b_diff |  -0.00528863  |       1.49471     |   0.0271801   |         0 |
|     44 | M0_Koopman      | test_iso_energy | T3       | S3b_std  |  -0.00389626  |       1.4961      |   0.0152873   |         0 |
|     44 | M0_Koopman      | test_ellip_peri | T3       | S2       |  -0.0153749   |       1.48463     |   0.526924    |         0 |
|     44 | M0_Koopman      | test_ellip_peri | T3       | S3a_diff |  -0.0171373   |       1.48286     |   0.622663    |         0 |
|     44 | M0_Koopman      | test_ellip_peri | T3       | S3b_diff |  -0.0170257   |       1.48297     |   0.630232    |         0 |
|     44 | M0_Koopman      | test_ellip_peri | T3       | S3b_std  |  -0.0161692   |       1.48383     |   0.547166    |         0 |
|     44 | M0_Koopman      | test_ellip_apo  | T3       | S2       |   0.0247121   |       1.52471     |   0.970119    |         0 |
|     44 | M0_Koopman      | test_ellip_apo  | T3       | S3a_diff |   0.0227666   |       1.52277     |   0.992971    |         0 |
|     44 | M0_Koopman      | test_ellip_apo  | T3       | S3b_diff |   0.0224968   |       1.5225      |   0.992822    |         0 |
|     44 | M0_Koopman      | test_ellip_apo  | T3       | S3b_std  |   0.0242587   |       1.52426     |   0.981155    |         0 |
|     44 | M1_circ         | test_id_ellip   | T1       | S2       |  -1.43349     |       0.0665089   |   0.993787    |         0 |
|     44 | M1_circ         | test_id_ellip   | T1       | S3a_diff |  -1.79723     |       0.297233    |   0.840348    |         0 |
|     44 | M1_circ         | test_id_ellip   | T1       | S3b_diff |  -1.50919     |       0.00919017  |   0.999338    |         0 |
|     44 | M1_circ         | test_id_ellip   | T1       | S3b_std  |  -1.50649     |       0.00649185  |   0.999814    |         0 |
|     44 | M1_circ         | test_id_ellip   | T3       | S2       |  -1.04949     |       0.450514    |   0.534876    |         0 |
|     44 | M1_circ         | test_id_ellip   | T3       | S3a_diff |  -1.25974     |       0.240258    |   0.571486    |         0 |
|     44 | M1_circ         | test_id_ellip   | T3       | S3b_diff |  -1.26151     |       0.238495    |   0.571503    |         0 |
|     44 | M1_circ         | test_id_ellip   | T3       | S3b_std  |  -1.25068     |       0.249324    |   0.570554    |         0 |
|     44 | M1_circ         | test_ood_ecc    | T1       | S2       |  -1.42843     |       0.0715698   |   0.994277    |         0 |
|     44 | M1_circ         | test_ood_ecc    | T1       | S3a_diff |  -1.50465     |       0.00464689  |   0.883705    |         0 |
|     44 | M1_circ         | test_ood_ecc    | T1       | S3b_diff |  -1.49055     |       0.00945469  |   0.998767    |         0 |
|     44 | M1_circ         | test_ood_ecc    | T1       | S3b_std  |  -1.496       |       0.00400091  |   0.999693    |         0 |
|     44 | M1_circ         | test_ood_ecc    | T3       | S2       |  -0.696124    |       0.803876    |   0.316525    |         0 |
|     44 | M1_circ         | test_ood_ecc    | T3       | S3a_diff |  -1.16559     |       0.334408    |   0.454584    |         0 |
|     44 | M1_circ         | test_ood_ecc    | T3       | S3b_diff |  -1.16659     |       0.333405    |   0.454792    |         0 |
|     44 | M1_circ         | test_ood_ecc    | T3       | S3b_std  |  -1.03444     |       0.465565    |   0.426652    |         0 |
|     44 | M1_circ         | test_iso_energy | T3       | S2       |  -1.14081     |       0.359195    |   0.51576     |         0 |
|     44 | M1_circ         | test_iso_energy | T3       | S3a_diff |  -1.46004     |       0.0399636   |   0.556829    |         0 |
|     44 | M1_circ         | test_iso_energy | T3       | S3b_diff |  -1.46142     |       0.0385844   |   0.556816    |         0 |
|     44 | M1_circ         | test_iso_energy | T3       | S3b_std  |  -1.41317     |       0.086829    |   0.555924    |         0 |
|     44 | M1_circ         | test_ellip_peri | T3       | S2       |  -0.506305    |       0.993695    |   0.363908    |         0 |
|     44 | M1_circ         | test_ellip_peri | T3       | S3a_diff |  -0.506843    |       0.993157    |   0.365402    |         0 |
|     44 | M1_circ         | test_ellip_peri | T3       | S3b_diff |  -0.506713    |       0.993287    |   0.365169    |         0 |
|     44 | M1_circ         | test_ellip_peri | T3       | S3b_std  |  -0.506058    |       0.993942    |   0.364409    |         0 |
|     44 | M1_circ         | test_ellip_apo  | T3       | S2       |  -0.794421    |       0.705579    |   0.828806    |         0 |
|     44 | M1_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.41265     |       0.0873454   |   0.999969    |         0 |
|     44 | M1_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.41428     |       0.0857217   |   0.999967    |         0 |
|     44 | M1_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.29282     |       0.207185    |   0.988311    |         0 |
|     44 | M2_circ         | test_id_ellip   | T1       | S2       |  -1.43447     |       0.0655279   |   0.993823    |         0 |
|     44 | M2_circ         | test_id_ellip   | T1       | S3a_diff |  -1.8159      |       0.3159      |   0.855851    |         0 |
|     44 | M2_circ         | test_id_ellip   | T1       | S3b_diff |  -1.50896     |       0.00895641  |   0.999408    |         0 |
|     44 | M2_circ         | test_id_ellip   | T1       | S3b_std  |  -1.5064      |       0.00640042  |   0.999823    |         0 |
|     44 | M2_circ         | test_id_ellip   | T3       | S2       |  -0.889874    |       0.610126    |   0.537332    |         0 |
|     44 | M2_circ         | test_id_ellip   | T3       | S3a_diff |  -1.06723     |       0.432767    |   0.560783    |         0 |
|     44 | M2_circ         | test_id_ellip   | T3       | S3b_diff |  -1.07126     |       0.428743    |   0.560681    |         0 |
|     44 | M2_circ         | test_id_ellip   | T3       | S3b_std  |  -1.07722     |       0.42278     |   0.560365    |         0 |
|     44 | M2_circ         | test_ood_ecc    | T1       | S2       |  -1.42968     |       0.0703222   |   0.994247    |         0 |
|     44 | M2_circ         | test_ood_ecc    | T1       | S3a_diff |  -1.63941     |       0.139409    |   0.893282    |         0 |
|     44 | M2_circ         | test_ood_ecc    | T1       | S3b_diff |  -1.49046     |       0.00953946  |   0.998778    |         0 |
|     44 | M2_circ         | test_ood_ecc    | T1       | S3b_std  |  -1.49623     |       0.00376776  |   0.999714    |         0 |
|     44 | M2_circ         | test_ood_ecc    | T3       | S2       |  -0.723083    |       0.776917    |   0.508121    |         0 |
|     44 | M2_circ         | test_ood_ecc    | T3       | S3a_diff |  -1.09765     |       0.40235     |   0.569586    |         0 |
|     44 | M2_circ         | test_ood_ecc    | T3       | S3b_diff |  -1.1022      |       0.3978      |   0.569611    |         0 |
|     44 | M2_circ         | test_ood_ecc    | T3       | S3b_std  |  -1.03654     |       0.463457    |   0.559079    |         0 |
|     44 | M2_circ         | test_iso_energy | T3       | S2       |  -1.10385     |       0.396149    |   0.587064    |         0 |
|     44 | M2_circ         | test_iso_energy | T3       | S3a_diff |  -1.38276     |       0.117239    |   0.602047    |         0 |
|     44 | M2_circ         | test_iso_energy | T3       | S3b_diff |  -1.3875      |       0.112495    |   0.602364    |         0 |
|     44 | M2_circ         | test_iso_energy | T3       | S3b_std  |  -1.36245     |       0.137552    |   0.605262    |         0 |
|     44 | M2_circ         | test_ellip_peri | T3       | S2       |  -0.0650077   |       1.43499     |   0.00994384  |         0 |
|     44 | M2_circ         | test_ellip_peri | T3       | S3a_diff |  -0.0669891   |       1.43301     |   0.0106076   |         0 |
|     44 | M2_circ         | test_ellip_peri | T3       | S3b_diff |  -0.0663468   |       1.43365     |   0.0103921   |         0 |
|     44 | M2_circ         | test_ellip_peri | T3       | S3b_std  |  -0.0649977   |       1.435       |   0.00996247  |         0 |
|     44 | M2_circ         | test_ellip_apo  | T3       | S2       |  -0.862317    |       0.637683    |   0.847752    |         0 |
|     44 | M2_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.39865     |       0.101354    |   0.999831    |         0 |
|     44 | M2_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.41156     |       0.0884382   |   0.999921    |         0 |
|     44 | M2_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.33757     |       0.162431    |   0.98816     |         0 |
|     44 | M3_circ         | test_id_ellip   | T1       | S2       |  -1.16715     |       0.332847    |   0.556455    |         0 |
|     44 | M3_circ         | test_id_ellip   | T1       | S3a_diff |  -0.394856    |       1.10514     |   0.036118    |         0 |
|     44 | M3_circ         | test_id_ellip   | T1       | S3b_diff |  -0.279852    |       1.22015     |   0.00397297  |         0 |
|     44 | M3_circ         | test_id_ellip   | T1       | S3b_std  |  -0.996799    |       0.503201    |   0.15544     |         0 |
|     44 | M3_circ         | test_id_ellip   | T3       | S2       |  -0.768379    |       0.731621    |   0.196442    |         0 |
|     44 | M3_circ         | test_id_ellip   | T3       | S3a_diff |  -0.892835    |       0.607165    |   0.180073    |         0 |
|     44 | M3_circ         | test_id_ellip   | T3       | S3b_diff |  -0.796684    |       0.703316    |   0.196539    |         0 |
|     44 | M3_circ         | test_id_ellip   | T3       | S3b_std  |  -0.90126     |       0.59874     |   0.214124    |         0 |
|     44 | M3_circ         | test_ood_ecc    | T1       | S2       |  -1.44238     |       0.0576157   |   0.993757    |         0 |
|     44 | M3_circ         | test_ood_ecc    | T1       | S3a_diff | nan           |     nan           | nan           |         4 |
|     44 | M3_circ         | test_ood_ecc    | T1       | S3b_diff |  -1.40445     |       0.095548    |   0.047462    |         0 |
|     44 | M3_circ         | test_ood_ecc    | T1       | S3b_std  |  -0.829709    |       0.670291    |   0.0894806   |         0 |
|     44 | M3_circ         | test_ood_ecc    | T3       | S2       |  -0.121745    |       1.37825     |   0.00476038  |         0 |
|     44 | M3_circ         | test_ood_ecc    | T3       | S3a_diff | nan           |     nan           | nan           |         1 |
|     44 | M3_circ         | test_ood_ecc    | T3       | S3b_diff |  -0.0179039   |       1.4821      |   6.80634e-05 |         0 |
|     44 | M3_circ         | test_ood_ecc    | T3       | S3b_std  |  -0.173372    |       1.32663     |   0.00676019  |         0 |
|     44 | M3_circ         | test_iso_energy | T3       | S2       |  -0.926627    |       0.573373    |   0.190462    |         0 |
|     44 | M3_circ         | test_iso_energy | T3       | S3a_diff | nan           |     nan           | nan           |         1 |
|     44 | M3_circ         | test_iso_energy | T3       | S3b_diff |  -1.2766      |       0.223401    |   0.0113083   |         0 |
|     44 | M3_circ         | test_iso_energy | T3       | S3b_std  |  -1.04161     |       0.458394    |   0.145247    |         0 |
|     44 | M3_circ         | test_ellip_peri | T3       | S2       |   0.28154     |       1.78154     |   0.0737232   |         0 |
|     44 | M3_circ         | test_ellip_peri | T3       | S3a_diff |   0.334842    |       1.83484     |   0.102912    |         0 |
|     44 | M3_circ         | test_ellip_peri | T3       | S3b_diff |   0.308106    |       1.80811     |   0.0860667   |         0 |
|     44 | M3_circ         | test_ellip_peri | T3       | S3b_std  |   0.285771    |       1.78577     |   0.0761663   |         0 |
|     44 | M3_circ         | test_ellip_apo  | T3       | S2       |  -1.6446      |       0.144597    |   0.6575      |         0 |
|     44 | M3_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.67225     |       0.172254    |   0.712077    |         0 |
|     44 | M3_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.83909     |       0.339091    |   0.652238    |         0 |
|     44 | M3_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.76289     |       0.262891    |   0.627443    |         0 |
|     44 | M1_Ellip_Short  | test_id_ellip   | T1       | S2       |  -1.43844     |       0.0615559   |   0.993832    |         0 |
|     44 | M1_Ellip_Short  | test_id_ellip   | T1       | S3a_diff |  -1.77006     |       0.270056    |   0.869683    |         0 |
|     44 | M1_Ellip_Short  | test_id_ellip   | T1       | S3b_diff |  -1.50643     |       0.00642813  |   0.999612    |         0 |
|     44 | M1_Ellip_Short  | test_id_ellip   | T1       | S3b_std  |  -1.50442     |       0.00441968  |   0.999887    |         0 |
|     44 | M1_Ellip_Short  | test_id_ellip   | T3       | S2       |  -1.43981     |       0.0601899   |   0.818576    |         0 |
|     44 | M1_Ellip_Short  | test_id_ellip   | T3       | S3a_diff |  -1.60483     |       0.104826    |   0.822267    |         0 |
|     44 | M1_Ellip_Short  | test_id_ellip   | T3       | S3b_diff |  -1.60544     |       0.105436    |   0.822313    |         0 |
|     44 | M1_Ellip_Short  | test_id_ellip   | T3       | S3b_std  |  -1.60989     |       0.109885    |   0.828272    |         0 |
|     44 | M1_Ellip_Short  | test_ood_ecc    | T1       | S2       |  -1.43856     |       0.061438    |   0.993886    |         0 |
|     44 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3a_diff |  -1.96842     |       0.468418    |   0.850119    |         0 |
|     44 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3b_diff |  -1.92366     |       0.423661    |   0.926559    |         0 |
|     44 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3b_std  |  -1.49772     |       0.00227676  |   0.999814    |         0 |
|     44 | M1_Ellip_Short  | test_ood_ecc    | T3       | S2       |  -1.52718     |       0.0271795   |   0.634889    |         0 |
|     44 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3a_diff |  -1.92883     |       0.428831    |   0.707265    |         0 |
|     44 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3b_diff |  -1.93558     |       0.435579    |   0.710064    |         0 |
|     44 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3b_std  |  -1.85448     |       0.354476    |   0.700829    |         0 |
|     44 | M1_Ellip_Short  | test_iso_energy | T3       | S2       |  -1.15129     |       0.348708    |   0.595314    |         0 |
|     44 | M1_Ellip_Short  | test_iso_energy | T3       | S3a_diff |  -1.40081     |       0.0991926   |   0.629106    |         0 |
|     44 | M1_Ellip_Short  | test_iso_energy | T3       | S3b_diff |  -1.40191     |       0.0980878   |   0.626983    |         0 |
|     44 | M1_Ellip_Short  | test_iso_energy | T3       | S3b_std  |  -1.37645     |       0.123552    |   0.632727    |         0 |
|     44 | M1_Ellip_Short  | test_ellip_peri | T3       | S2       |  -0.534517    |       0.965483    |   0.986546    |         0 |
|     44 | M1_Ellip_Short  | test_ellip_peri | T3       | S3a_diff |  -0.499207    |       1.00079     |   0.964996    |         0 |
|     44 | M1_Ellip_Short  | test_ellip_peri | T3       | S3b_diff |  -0.524483    |       0.975517    |   0.977249    |         0 |
|     44 | M1_Ellip_Short  | test_ellip_peri | T3       | S3b_std  |  -0.524173    |       0.975827    |   0.975398    |         0 |
|     44 | M1_Ellip_Short  | test_ellip_apo  | T3       | S2       |  -1.56389     |       0.0638883   |   0.954071    |         0 |
|     44 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3a_diff |  -1.82131     |       0.321313    |   0.999211    |         0 |
|     44 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3b_diff |  -1.82356     |       0.323556    |   0.999143    |         0 |
|     44 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3b_std  |  -1.81697     |       0.316969    |   0.998207    |         0 |
|     44 | M1_Ellip_Inv    | test_id_ellip   | T1       | S2       |  -1.43554     |       0.064456    |   0.993871    |         0 |
|     44 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3a_diff |  -1.69984     |       0.199844    |   0.829216    |         0 |
|     44 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3b_diff |  -1.5622      |       0.0622031   |   0.986918    |         0 |
|     44 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3b_std  |  -1.50699     |       0.0069876   |   0.999813    |         0 |
|     44 | M1_Ellip_Inv    | test_id_ellip   | T3       | S2       |  -1.46524     |       0.0347582   |   0.987127    |         0 |
|     44 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3a_diff |  -1.5418      |       0.0418009   |   0.993366    |         0 |
|     44 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3b_diff |  -1.54305     |       0.0430543   |   0.99333     |         0 |
|     44 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3b_std  |  -1.5498      |       0.0497986   |   0.992664    |         0 |
|     44 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S2       |  -1.43464     |       0.0653613   |   0.993985    |         0 |
|     44 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3a_diff |  -2.34968     |       0.84968     |   0.84658     |         0 |
|     44 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3b_diff |  -2.22626     |       0.72626     |   0.971678    |         0 |
|     44 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3b_std  |  -1.74122     |       0.241221    |   0.936491    |         0 |
|     44 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S2       |  -1.40376     |       0.0962383   |   0.961732    |         0 |
|     44 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3a_diff |  -1.46744     |       0.0325553   |   0.958535    |         0 |
|     44 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3b_diff |  -1.4706      |       0.0294025   |   0.957859    |         0 |
|     44 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3b_std  |  -1.47265     |       0.0273546   |   0.956494    |         0 |
|     44 | M1_Ellip_Inv    | test_iso_energy | T3       | S2       |  -1.38877     |       0.111233    |   0.991459    |         0 |
|     44 | M1_Ellip_Inv    | test_iso_energy | T3       | S3a_diff |  -1.50143     |       0.00143104  |   0.984926    |         0 |
|     44 | M1_Ellip_Inv    | test_iso_energy | T3       | S3b_diff |  -1.50333     |       0.00333411  |   0.984734    |         0 |
|     44 | M1_Ellip_Inv    | test_iso_energy | T3       | S3b_std  |  -1.51648     |       0.0164775   |   0.981743    |         0 |
|     44 | M1_Ellip_Inv    | test_ellip_peri | T3       | S2       |  -1.53615     |       0.0361504   |   0.983083    |         0 |
|     44 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3a_diff |  -1.62755     |       0.127548    |   0.998072    |         0 |
|     44 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3b_diff |  -1.62816     |       0.12816     |   0.998073    |         0 |
|     44 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3b_std  |  -1.63441     |       0.134406    |   0.998444    |         0 |
|     44 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S2       |  -1.43967     |       0.0603339   |   0.991742    |         0 |
|     44 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3a_diff |  -1.52996     |       0.0299576   |   0.99921     |         0 |
|     44 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3b_diff |  -1.53308     |       0.0330764   |   0.999067    |         0 |
|     44 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3b_std  |  -1.54331     |       0.043314    |   0.998116    |         0 |
|     44 | M2_Ellip_Inv    | test_id_ellip   | T1       | S2       |  -1.43755     |       0.0624527   |   0.993744    |         0 |
|     44 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3a_diff |  -1.65282     |       0.15282     |   0.853372    |         0 |
|     44 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3b_diff |  -1.59693     |       0.0969302   |   0.9667      |         0 |
|     44 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3b_std  |  -1.5076      |       0.00759809  |   0.99981     |         0 |
|     44 | M2_Ellip_Inv    | test_id_ellip   | T3       | S2       |  -1.46406     |       0.0359397   |   0.989697    |         0 |
|     44 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3a_diff |  -1.54706     |       0.0470638   |   0.998053    |         0 |
|     44 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3b_diff |  -1.54759     |       0.0475875   |   0.998047    |         0 |
|     44 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3b_std  |  -1.55396     |       0.0539647   |   0.997978    |         0 |
|     44 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S2       |  -1.43796     |       0.0620365   |   0.993424    |         0 |
|     44 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3a_diff | nan           |     nan           | nan           |         6 |
|     44 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3b_diff |  -2.14691     |       0.646907    |   0.928107    |         0 |
|     44 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3b_std  |  -1.8371      |       0.337097    |   0.919358    |         0 |
|     44 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S2       |  -1.34882     |       0.151177    |   0.96412     |         0 |
|     44 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3a_diff |  -1.39455     |       0.105454    |   0.95887     |         0 |
|     44 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3b_diff |  -1.39551     |       0.104488    |   0.958589    |         0 |
|     44 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3b_std  |  -1.39367     |       0.106327    |   0.957503    |         0 |
|     44 | M2_Ellip_Inv    | test_iso_energy | T3       | S2       |  -1.40489     |       0.0951127   |   0.994927    |         0 |
|     44 | M2_Ellip_Inv    | test_iso_energy | T3       | S3a_diff |  -1.52733     |       0.0273281   |   0.991399    |         0 |
|     44 | M2_Ellip_Inv    | test_iso_energy | T3       | S3b_diff |  -1.52817     |       0.0281688   |   0.99134     |         0 |
|     44 | M2_Ellip_Inv    | test_iso_energy | T3       | S3b_std  |  -1.53999     |       0.039994    |   0.989669    |         0 |
|     44 | M2_Ellip_Inv    | test_ellip_peri | T3       | S2       |  -1.5566      |       0.0565984   |   0.986733    |         0 |
|     44 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3a_diff |  -1.61535     |       0.115347    |   0.997621    |         0 |
|     44 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3b_diff |  -1.61582     |       0.115825    |   0.99763     |         0 |
|     44 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3b_std  |  -1.62008     |       0.120083    |   0.998031    |         0 |
|     44 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S2       |  -1.40623     |       0.0937682   |   0.995298    |         0 |
|     44 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3a_diff |  -1.45208     |       0.0479181   |   0.999765    |         0 |
|     44 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3b_diff |  -1.45266     |       0.047338    |   0.999753    |         0 |
|     44 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3b_std  |  -1.45528     |       0.0447241   |   0.999282    |         0 |
|     44 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S2       |  -1.30534     |       0.19466     |   0.722025    |         0 |
|     44 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3a_diff | nan           |     nan           | nan           |         3 |
|     44 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3b_diff |  -1.72845     |       0.228453    |   0.188073    |         0 |
|     44 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3b_std  |  -0.689508    |       0.810492    |   0.0489742   |         0 |
|     44 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S2       |  -3.24036e-05 |       1.49997     |   0.0200743   |         0 |
|     44 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3a_diff |  -0.0472459   |       1.45275     |   0.0232186   |         0 |
|     44 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3b_diff |  -0.0455242   |       1.45448     |   0.0236981   |         0 |
|     44 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3b_std  |  -0.00396484  |       1.49604     |   0.0168323   |         0 |
|     44 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S2       |  -1.45214     |       0.0478643   |   0.992465    |         0 |
|     44 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3a_diff | nan           |     nan           | nan           |         6 |
|     44 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3b_diff |  -1.13023     |       0.369767    |   0.0556394   |         0 |
|     44 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3b_std  |  -0.538109    |       0.961891    |   0.0247714   |         0 |
|     44 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S2       |  -3.18478e-06 |       1.5         |   0.000171664 |         0 |
|     44 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3a_diff |  -0.0196099   |       1.48039     |   0.00387224  |         0 |
|     44 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3b_diff |  -0.0165505   |       1.48345     |   0.00290037  |         0 |
|     44 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3b_std  |  -0.000440533 |       1.49956     |   0.000197203 |         0 |
|     44 | M3_Ellip_SIGReg | test_iso_energy | T3       | S2       |  -7.05462e-05 |       1.49993     |   0.0994072   |         0 |
|     44 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3a_diff |  -0.0857894   |       1.41421     |   0.094507    |         0 |
|     44 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3b_diff |  -0.080482    |       1.41952     |   0.0905477   |         0 |
|     44 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3b_std  |  -0.0104434   |       1.48956     |   0.119063    |         0 |
|     44 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S2       |   3.35608e-06 |       1.5         |   0.000306708 |         0 |
|     44 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3a_diff |   0.0187371   |       1.51874     |   0.00620898  |         0 |
|     44 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3b_diff |   0.0142649   |       1.51426     |   0.00369548  |         0 |
|     44 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3b_std  |   0.000859115 |       1.50086     |   0.00123613  |         0 |
|     44 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S2       |   4.71955e-05 |       1.50005     |   0.128724    |         0 |
|     44 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3a_diff |   0.0362336   |       1.53623     |   0.0450358   |         0 |
|     44 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3b_diff |   0.0414195   |       1.54142     |   0.0657722   |         0 |
|     44 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3b_std  |   0.00560419  |       1.5056      |   0.0992473   |         0 |
|     45 | M0_Koopman      | test_id_ellip   | T1       | S2       |  -1.44196     |       0.0580386   |   0.993115    |         0 |
|     45 | M0_Koopman      | test_id_ellip   | T1       | S3a_diff |  -1.79149     |       0.291487    |   0.81639     |         0 |
|     45 | M0_Koopman      | test_id_ellip   | T1       | S3b_diff |  -1.5018      |       0.00179771  |   0.99957     |         0 |
|     45 | M0_Koopman      | test_id_ellip   | T1       | S3b_std  |  -1.50341     |       0.00341488  |   0.999696    |         0 |
|     45 | M0_Koopman      | test_id_ellip   | T3       | S2       |   0.008049    |       1.50805     |   0.0549385   |         0 |
|     45 | M0_Koopman      | test_id_ellip   | T3       | S3a_diff |   0.00537672  |       1.50538     |   0.0251607   |         0 |
|     45 | M0_Koopman      | test_id_ellip   | T3       | S3b_diff |   0.00561874  |       1.50562     |   0.0275546   |         0 |
|     45 | M0_Koopman      | test_id_ellip   | T3       | S3b_std  |   0.00738867  |       1.50739     |   0.0466963   |         0 |
|     45 | M0_Koopman      | test_ood_ecc    | T1       | S2       |  -1.42845     |       0.071546    |   0.992355    |         0 |
|     45 | M0_Koopman      | test_ood_ecc    | T1       | S3a_diff |  -1.52517     |       0.0251685   |   0.882441    |         0 |
|     45 | M0_Koopman      | test_ood_ecc    | T1       | S3b_diff |  -1.50555     |       0.00554994  |   0.997608    |         0 |
|     45 | M0_Koopman      | test_ood_ecc    | T1       | S3b_std  |  -1.4992      |       0.000795373 |   0.999378    |         0 |
|     45 | M0_Koopman      | test_ood_ecc    | T3       | S2       |   0.0104852   |       1.51049     |   0.0375391   |         0 |
|     45 | M0_Koopman      | test_ood_ecc    | T3       | S3a_diff |   0.0109552   |       1.51096     |   0.0481306   |         0 |
|     45 | M0_Koopman      | test_ood_ecc    | T3       | S3b_diff |   0.0105857   |       1.51059     |   0.0460103   |         0 |
|     45 | M0_Koopman      | test_ood_ecc    | T3       | S3b_std  |   0.0111203   |       1.51112     |   0.044125    |         0 |
|     45 | M0_Koopman      | test_iso_energy | T3       | S2       |   0.0186411   |       1.51864     |   0.158328    |         0 |
|     45 | M0_Koopman      | test_iso_energy | T3       | S3a_diff |   0.0146187   |       1.51462     |   0.120058    |         0 |
|     45 | M0_Koopman      | test_iso_energy | T3       | S3b_diff |   0.0140745   |       1.51407     |   0.119505    |         0 |
|     45 | M0_Koopman      | test_iso_energy | T3       | S3b_std  |   0.0178021   |       1.5178      |   0.146339    |         0 |
|     45 | M0_Koopman      | test_ellip_peri | T3       | S2       |  -0.024162    |       1.47584     |   0.915499    |         0 |
|     45 | M0_Koopman      | test_ellip_peri | T3       | S3a_diff |  -0.0286428   |       1.47136     |   0.919298    |         0 |
|     45 | M0_Koopman      | test_ellip_peri | T3       | S3b_diff |  -0.0282999   |       1.4717      |   0.923118    |         0 |
|     45 | M0_Koopman      | test_ellip_peri | T3       | S3b_std  |  -0.0259006   |       1.4741      |   0.918828    |         0 |
|     45 | M0_Koopman      | test_ellip_apo  | T3       | S2       |   0.0264692   |       1.52647     |   0.98726     |         0 |
|     45 | M0_Koopman      | test_ellip_apo  | T3       | S3a_diff |   0.0283669   |       1.52837     |   0.971439    |         0 |
|     45 | M0_Koopman      | test_ellip_apo  | T3       | S3b_diff |   0.0280351   |       1.52804     |   0.973844    |         0 |
|     45 | M0_Koopman      | test_ellip_apo  | T3       | S3b_std  |   0.0283128   |       1.52831     |   0.985491    |         0 |
|     45 | M1_circ         | test_id_ellip   | T1       | S2       |  -1.4425      |       0.0575028   |   0.993033    |         0 |
|     45 | M1_circ         | test_id_ellip   | T1       | S3a_diff |  -1.78356     |       0.28356     |   0.818729    |         0 |
|     45 | M1_circ         | test_id_ellip   | T1       | S3b_diff |  -1.50136     |       0.00136463  |   0.999561    |         0 |
|     45 | M1_circ         | test_id_ellip   | T1       | S3b_std  |  -1.50318     |       0.00318169  |   0.999694    |         0 |
|     45 | M1_circ         | test_id_ellip   | T3       | S2       |  -1.01351     |       0.486493    |   0.519798    |         0 |
|     45 | M1_circ         | test_id_ellip   | T3       | S3a_diff |  -1.26145     |       0.238554    |   0.522415    |         0 |
|     45 | M1_circ         | test_id_ellip   | T3       | S3b_diff |  -1.26266     |       0.237337    |   0.522181    |         0 |
|     45 | M1_circ         | test_id_ellip   | T3       | S3b_std  |  -1.22471     |       0.275287    |   0.524131    |         0 |
|     45 | M1_circ         | test_ood_ecc    | T1       | S2       |  -1.42863     |       0.0713694   |   0.992332    |         0 |
|     45 | M1_circ         | test_ood_ecc    | T1       | S3a_diff |  -1.53209     |       0.0320944   |   0.886177    |         0 |
|     45 | M1_circ         | test_ood_ecc    | T1       | S3b_diff |  -1.50541     |       0.00541368  |   0.997661    |         0 |
|     45 | M1_circ         | test_ood_ecc    | T1       | S3b_std  |  -1.49914     |       0.000861802 |   0.999386    |         0 |
|     45 | M1_circ         | test_ood_ecc    | T3       | S2       |  -0.617897    |       0.882103    |   0.226819    |         0 |
|     45 | M1_circ         | test_ood_ecc    | T3       | S3a_diff |  -1.12535     |       0.374651    |   0.345122    |         0 |
|     45 | M1_circ         | test_ood_ecc    | T3       | S3b_diff |  -1.12473     |       0.375268    |   0.344986    |         0 |
|     45 | M1_circ         | test_ood_ecc    | T3       | S3b_std  |  -0.932627    |       0.567373    |   0.310057    |         0 |
|     45 | M1_circ         | test_iso_energy | T3       | S2       |  -1.31746     |       0.182543    |   0.715108    |         0 |
|     45 | M1_circ         | test_iso_energy | T3       | S3a_diff |  -1.67574     |       0.175744    |   0.751416    |         0 |
|     45 | M1_circ         | test_iso_energy | T3       | S3b_diff |  -1.67688     |       0.176877    |   0.751147    |         0 |
|     45 | M1_circ         | test_iso_energy | T3       | S3b_std  |  -1.62942     |       0.129415    |   0.754165    |         0 |
|     45 | M1_circ         | test_ellip_peri | T3       | S2       |  -0.592961    |       0.907039    |   0.551253    |         0 |
|     45 | M1_circ         | test_ellip_peri | T3       | S3a_diff |  -0.591734    |       0.908266    |   0.551311    |         0 |
|     45 | M1_circ         | test_ellip_peri | T3       | S3b_diff |  -0.5923      |       0.9077      |   0.551753    |         0 |
|     45 | M1_circ         | test_ellip_peri | T3       | S3b_std  |  -0.591041    |       0.908959    |   0.550507    |         0 |
|     45 | M1_circ         | test_ellip_apo  | T3       | S2       |  -0.809835    |       0.690165    |   0.833315    |         0 |
|     45 | M1_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.41109     |       0.088912    |   0.999985    |         0 |
|     45 | M1_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.41515     |       0.0848468   |   0.999986    |         0 |
|     45 | M1_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.30192     |       0.198083    |   0.988744    |         0 |
|     45 | M2_circ         | test_id_ellip   | T1       | S2       |  -1.4445      |       0.0555002   |   0.992921    |         0 |
|     45 | M2_circ         | test_id_ellip   | T1       | S3a_diff |  -1.8106      |       0.3106      |   0.83497     |         0 |
|     45 | M2_circ         | test_id_ellip   | T1       | S3b_diff |  -1.50142     |       0.0014222   |   0.999591    |         0 |
|     45 | M2_circ         | test_id_ellip   | T1       | S3b_std  |  -1.50321     |       0.00320983  |   0.999702    |         0 |
|     45 | M2_circ         | test_id_ellip   | T3       | S2       |  -1.04546     |       0.454537    |   0.556111    |         0 |
|     45 | M2_circ         | test_id_ellip   | T3       | S3a_diff |  -1.22845     |       0.271548    |   0.552943    |         0 |
|     45 | M2_circ         | test_id_ellip   | T3       | S3b_diff |  -1.23478     |       0.265223    |   0.552701    |         0 |
|     45 | M2_circ         | test_id_ellip   | T3       | S3b_std  |  -1.22728     |       0.27272     |   0.553926    |         0 |
|     45 | M2_circ         | test_ood_ecc    | T1       | S2       |  -1.43323     |       0.0667704   |   0.99244     |         0 |
|     45 | M2_circ         | test_ood_ecc    | T1       | S3a_diff |  -1.7166      |       0.2166      |   0.889888    |         0 |
|     45 | M2_circ         | test_ood_ecc    | T1       | S3b_diff |  -1.50568     |       0.00567981  |   0.997876    |         0 |
|     45 | M2_circ         | test_ood_ecc    | T1       | S3b_std  |  -1.5003      |       0.000303084 |   0.999488    |         0 |
|     45 | M2_circ         | test_ood_ecc    | T3       | S2       |  -0.42893     |       1.07107     |   0.112492    |         0 |
|     45 | M2_circ         | test_ood_ecc    | T3       | S3a_diff |  -0.812949    |       0.687051    |   0.226216    |         0 |
|     45 | M2_circ         | test_ood_ecc    | T3       | S3b_diff |  -0.815714    |       0.684286    |   0.226351    |         0 |
|     45 | M2_circ         | test_ood_ecc    | T3       | S3b_std  |  -0.71351     |       0.78649     |   0.198638    |         0 |
|     45 | M2_circ         | test_iso_energy | T3       | S2       |  -1.273       |       0.226996    |   0.686827    |         0 |
|     45 | M2_circ         | test_iso_energy | T3       | S3a_diff |  -1.58019     |       0.0801858   |   0.735529    |         0 |
|     45 | M2_circ         | test_iso_energy | T3       | S3b_diff |  -1.58561     |       0.0856104   |   0.735306    |         0 |
|     45 | M2_circ         | test_iso_energy | T3       | S3b_std  |  -1.57911     |       0.0791092   |   0.737979    |         0 |
|     45 | M2_circ         | test_ellip_peri | T3       | S2       |  -0.136356    |       1.36364     |   0.0435652   |         0 |
|     45 | M2_circ         | test_ellip_peri | T3       | S3a_diff |  -0.138875    |       1.36112     |   0.0454801   |         0 |
|     45 | M2_circ         | test_ellip_peri | T3       | S3b_diff |  -0.13849     |       1.36151     |   0.0451399   |         0 |
|     45 | M2_circ         | test_ellip_peri | T3       | S3b_std  |  -0.13612     |       1.36388     |   0.0435007   |         0 |
|     45 | M2_circ         | test_ellip_apo  | T3       | S2       |  -0.727978    |       0.772022    |   0.806723    |         0 |
|     45 | M2_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.22934     |       0.270664    |   0.997784    |         0 |
|     45 | M2_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.2431      |       0.256898    |   0.99841     |         0 |
|     45 | M2_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.19331     |       0.306688    |   0.98663     |         0 |
|     45 | M3_circ         | test_id_ellip   | T1       | S2       |  -1.35983     |       0.140168    |   0.683492    |         0 |
|     45 | M3_circ         | test_id_ellip   | T1       | S3a_diff | nan           |     nan           | nan           |         4 |
|     45 | M3_circ         | test_id_ellip   | T1       | S3b_diff |  -0.914574    |       0.585426    |   0.0385422   |         0 |
|     45 | M3_circ         | test_id_ellip   | T1       | S3b_std  |   2.6568      |       4.1568      |   0.196266    |         0 |
|     45 | M3_circ         | test_id_ellip   | T3       | S2       |  -0.22481     |       1.27519     |   0.0153946   |         0 |
|     45 | M3_circ         | test_id_ellip   | T3       | S3a_diff |  -0.373583    |       1.12642     |   0.0266488   |         0 |
|     45 | M3_circ         | test_id_ellip   | T3       | S3b_diff |  -0.44293     |       1.05707     |   0.0384295   |         0 |
|     45 | M3_circ         | test_id_ellip   | T3       | S3b_std  |  -0.492277    |       1.00772     |   0.04776     |         0 |
|     45 | M3_circ         | test_ood_ecc    | T1       | S2       |  -1.44478     |       0.0552241   |   0.993368    |         0 |
|     45 | M3_circ         | test_ood_ecc    | T1       | S3a_diff | nan           |     nan           | nan           |         7 |
|     45 | M3_circ         | test_ood_ecc    | T1       | S3b_diff |   0.689389    |       2.18939     |   0.0250632   |         0 |
|     45 | M3_circ         | test_ood_ecc    | T1       | S3b_std  |   1.8364      |       3.3364      |   0.155288    |         0 |
|     45 | M3_circ         | test_ood_ecc    | T3       | S2       |  -0.263131    |       1.23687     |   0.0219168   |         0 |
|     45 | M3_circ         | test_ood_ecc    | T3       | S3a_diff | nan           |     nan           | nan           |         2 |
|     45 | M3_circ         | test_ood_ecc    | T3       | S3b_diff |  -0.159221    |       1.34078     |   0.000152462 |         0 |
|     45 | M3_circ         | test_ood_ecc    | T3       | S3b_std  |  -0.214323    |       1.28568     |   0.0110412   |         0 |
|     45 | M3_circ         | test_iso_energy | T3       | S2       |  -0.983718    |       0.516282    |   0.280775    |         0 |
|     45 | M3_circ         | test_iso_energy | T3       | S3a_diff | nan           |     nan           | nan           |         1 |
|     45 | M3_circ         | test_iso_energy | T3       | S3b_diff |  -1.29696     |       0.203037    |   0.21052     |         0 |
|     45 | M3_circ         | test_iso_energy | T3       | S3b_std  |  -1.15736     |       0.342637    |   0.266585    |         0 |
|     45 | M3_circ         | test_ellip_peri | T3       | S2       |  -0.218371    |       1.28163     |   0.232089    |         0 |
|     45 | M3_circ         | test_ellip_peri | T3       | S3a_diff |  -0.204628    |       1.29537     |   0.217673    |         0 |
|     45 | M3_circ         | test_ellip_peri | T3       | S3b_diff |  -0.21535     |       1.28465     |   0.231779    |         0 |
|     45 | M3_circ         | test_ellip_peri | T3       | S3b_std  |  -0.215987    |       1.28401     |   0.231094    |         0 |
|     45 | M3_circ         | test_ellip_apo  | T3       | S2       |  -0.589106    |       0.910894    |   0.425608    |         0 |
|     45 | M3_circ         | test_ellip_apo  | T3       | S3a_diff |  -0.678664    |       0.821336    |   0.552732    |         0 |
|     45 | M3_circ         | test_ellip_apo  | T3       | S3b_diff |  -0.645359    |       0.854641    |   0.496916    |         0 |
|     45 | M3_circ         | test_ellip_apo  | T3       | S3b_std  |  -0.597361    |       0.902639    |   0.443514    |         0 |
|     45 | M1_Ellip_Short  | test_id_ellip   | T1       | S2       |  -1.44608     |       0.0539183   |   0.993048    |         0 |
|     45 | M1_Ellip_Short  | test_id_ellip   | T1       | S3a_diff |  -1.79113     |       0.291127    |   0.83132     |         0 |
|     45 | M1_Ellip_Short  | test_id_ellip   | T1       | S3b_diff |  -1.56862     |       0.0686181   |   0.976129    |         0 |
|     45 | M1_Ellip_Short  | test_id_ellip   | T1       | S3b_std  |  -1.50369     |       0.00368896  |   0.999757    |         0 |
|     45 | M1_Ellip_Short  | test_id_ellip   | T3       | S2       |  -1.26482     |       0.235182    |   0.748908    |         0 |
|     45 | M1_Ellip_Short  | test_id_ellip   | T3       | S3a_diff |  -1.47003     |       0.0299663   |   0.742167    |         0 |
|     45 | M1_Ellip_Short  | test_id_ellip   | T3       | S3b_diff |  -1.47577     |       0.0242298   |   0.742556    |         0 |
|     45 | M1_Ellip_Short  | test_id_ellip   | T3       | S3b_std  |  -1.48945     |       0.0105499   |   0.744061    |         0 |
|     45 | M1_Ellip_Short  | test_ood_ecc    | T1       | S2       |  -1.44056     |       0.0594445   |   0.993046    |         0 |
|     45 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3a_diff |  -1.95105     |       0.451053    |   0.844798    |         0 |
|     45 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3b_diff |  -2.15044     |       0.650437    |   0.94139     |         0 |
|     45 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3b_std  |  -1.50233     |       0.00233207  |   0.999739    |         0 |
|     45 | M1_Ellip_Short  | test_ood_ecc    | T3       | S2       |  -0.980476    |       0.519524    |   0.299594    |         0 |
|     45 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3a_diff |  -1.53732     |       0.0373233   |   0.438898    |         0 |
|     45 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3b_diff |  -1.55068     |       0.0506807   |   0.440622    |         0 |
|     45 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3b_std  |  -1.42269     |       0.0773065   |   0.413647    |         0 |
|     45 | M1_Ellip_Short  | test_iso_energy | T3       | S2       |  -1.2217      |       0.278298    |   0.729079    |         0 |
|     45 | M1_Ellip_Short  | test_iso_energy | T3       | S3a_diff |  -1.49596     |       0.00403798  |   0.775636    |         0 |
|     45 | M1_Ellip_Short  | test_iso_energy | T3       | S3b_diff |  -1.50097     |       0.000969833 |   0.774755    |         0 |
|     45 | M1_Ellip_Short  | test_iso_energy | T3       | S3b_std  |  -1.49311     |       0.00688933  |   0.78042     |         0 |
|     45 | M1_Ellip_Short  | test_ellip_peri | T3       | S2       |  -0.791642    |       0.708358    |   0.989547    |         0 |
|     45 | M1_Ellip_Short  | test_ellip_peri | T3       | S3a_diff |  -0.694394    |       0.805606    |   0.959577    |         0 |
|     45 | M1_Ellip_Short  | test_ellip_peri | T3       | S3b_diff |  -0.755149    |       0.744851    |   0.977767    |         0 |
|     45 | M1_Ellip_Short  | test_ellip_peri | T3       | S3b_std  |  -0.756572    |       0.743428    |   0.980801    |         0 |
|     45 | M1_Ellip_Short  | test_ellip_apo  | T3       | S2       |  -1.58648     |       0.0864822   |   0.946914    |         0 |
|     45 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3a_diff |  -1.91654     |       0.416536    |   0.999229    |         0 |
|     45 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3b_diff |  -1.92106     |       0.421055    |   0.99918     |         0 |
|     45 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3b_std  |  -1.89706     |       0.397056    |   0.997017    |         0 |
|     45 | M1_Ellip_Inv    | test_id_ellip   | T1       | S2       |  -1.44491     |       0.0550916   |   0.992652    |         0 |
|     45 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3a_diff |  -1.67144     |       0.171444    |   0.852451    |         0 |
|     45 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3b_diff |  -1.50076     |       0.000758254 |   0.999504    |         0 |
|     45 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3b_std  |  -1.50232     |       0.00232335  |   0.999721    |         0 |
|     45 | M1_Ellip_Inv    | test_id_ellip   | T3       | S2       |  -1.44445     |       0.0555528   |   0.990397    |         0 |
|     45 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3a_diff |  -1.53683     |       0.0368267   |   0.995977    |         0 |
|     45 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3b_diff |  -1.53736     |       0.0373596   |   0.995945    |         0 |
|     45 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3b_std  |  -1.54616     |       0.0461622   |   0.995544    |         0 |
|     45 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S2       |  -1.43562     |       0.0643773   |   0.992974    |         0 |
|     45 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3a_diff |  -2.34083     |       0.84083     |   0.832716    |         0 |
|     45 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3b_diff |  -2.15689     |       0.656891    |   0.941419    |         0 |
|     45 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3b_std  |  -1.75084     |       0.250837    |   0.938281    |         0 |
|     45 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S2       |  -1.07683     |       0.423174    |   0.725202    |         0 |
|     45 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3a_diff |  -1.14107     |       0.358934    |   0.739866    |         0 |
|     45 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3b_diff |  -1.14338     |       0.356624    |   0.740587    |         0 |
|     45 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3b_std  |  -1.14141     |       0.358586    |   0.73813     |         0 |
|     45 | M1_Ellip_Inv    | test_iso_energy | T3       | S2       |  -1.40546     |       0.0945378   |   0.993257    |         0 |
|     45 | M1_Ellip_Inv    | test_iso_energy | T3       | S3a_diff |  -1.5077      |       0.00769886  |   0.991969    |         0 |
|     45 | M1_Ellip_Inv    | test_iso_energy | T3       | S3b_diff |  -1.5088      |       0.00879535  |   0.992002    |         0 |
|     45 | M1_Ellip_Inv    | test_iso_energy | T3       | S3b_std  |  -1.5175      |       0.0174978   |   0.990706    |         0 |
|     45 | M1_Ellip_Inv    | test_ellip_peri | T3       | S2       |  -1.37643     |       0.123565    |   0.990654    |         0 |
|     45 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3a_diff |  -1.42375     |       0.0762492   |   0.998763    |         0 |
|     45 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3b_diff |  -1.42401     |       0.0759928   |   0.998765    |         0 |
|     45 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3b_std  |  -1.4264      |       0.0736018   |   0.998821    |         0 |
|     45 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S2       |  -1.46077     |       0.0392292   |   0.992125    |         0 |
|     45 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3a_diff |  -1.5506      |       0.0506048   |   0.999014    |         0 |
|     45 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3b_diff |  -1.55196     |       0.0519604   |   0.99895     |         0 |
|     45 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3b_std  |  -1.55896     |       0.0589649   |   0.998297    |         0 |
|     45 | M2_Ellip_Inv    | test_id_ellip   | T1       | S2       |  -1.44594     |       0.0540554   |   0.992332    |         0 |
|     45 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3a_diff |  -1.61145     |       0.111451    |   0.851625    |         0 |
|     45 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3b_diff |  -1.6039      |       0.103898    |   0.977881    |         0 |
|     45 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3b_std  |  -1.5012      |       0.00119604  |   0.999691    |         0 |
|     45 | M2_Ellip_Inv    | test_id_ellip   | T3       | S2       |  -1.43834     |       0.061662    |   0.991535    |         0 |
|     45 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3a_diff |  -1.52852     |       0.0285218   |   0.996478    |         0 |
|     45 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3b_diff |  -1.52883     |       0.0288288   |   0.996459    |         0 |
|     45 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3b_std  |  -1.53613     |       0.0361297   |   0.995988    |         0 |
|     45 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S2       |  -1.43395     |       0.0660527   |   0.99353     |         0 |
|     45 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3a_diff |  -2.60584     |       1.10584     |   0.872524    |         0 |
|     45 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3b_diff |  -2.19845     |       0.698451    |   0.96141     |         0 |
|     45 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3b_std  |  -1.84693     |       0.34693     |   0.936808    |         0 |
|     45 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S2       |  -1.08392     |       0.41608     |   0.749818    |         0 |
|     45 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3a_diff |  -1.09905     |       0.400953    |   0.750451    |         0 |
|     45 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3b_diff |  -1.10032     |       0.399683    |   0.751266    |         0 |
|     45 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3b_std  |  -1.09698     |       0.403021    |   0.745159    |         0 |
|     45 | M2_Ellip_Inv    | test_iso_energy | T3       | S2       |  -1.3739      |       0.126102    |   0.990338    |         0 |
|     45 | M2_Ellip_Inv    | test_iso_energy | T3       | S3a_diff |  -1.48405     |       0.015952    |   0.987534    |         0 |
|     45 | M2_Ellip_Inv    | test_iso_energy | T3       | S3b_diff |  -1.4845      |       0.0154977   |   0.98751     |         0 |
|     45 | M2_Ellip_Inv    | test_iso_energy | T3       | S3b_std  |  -1.49354     |       0.0064553   |   0.985873    |         0 |
|     45 | M2_Ellip_Inv    | test_ellip_peri | T3       | S2       |  -1.46576     |       0.0342447   |   0.988882    |         0 |
|     45 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3a_diff |  -1.5264      |       0.0263974   |   0.998552    |         0 |
|     45 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3b_diff |  -1.52657     |       0.0265748   |   0.998554    |         0 |
|     45 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3b_std  |  -1.53012     |       0.0301183   |   0.998527    |         0 |
|     45 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S2       |  -1.44295     |       0.0570492   |   0.991036    |         0 |
|     45 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3a_diff |  -1.52509     |       0.0250942   |   0.999787    |         0 |
|     45 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3b_diff |  -1.52613     |       0.0261339   |   0.999775    |         0 |
|     45 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3b_std  |  -1.53326     |       0.0332566   |   0.999413    |         0 |
|     45 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S2       |  -1.2403      |       0.259702    |   0.687449    |         0 |
|     45 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3a_diff | nan           |     nan           | nan           |         1 |
|     45 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3b_diff |   0.0813459   |       1.58135     |   0.000248045 |         0 |
|     45 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3b_std  |  -1.38073     |       0.119274    |   0.082025    |         0 |
|     45 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S2       |  -4.01757e-05 |       1.49996     |   0.0714911   |         0 |
|     45 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3a_diff |  -0.055982    |       1.44402     |   0.0658555   |         0 |
|     45 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3b_diff |  -0.0561998   |       1.4438      |   0.0712909   |         0 |
|     45 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3b_std  |  -0.0045817   |       1.49542     |   0.060912    |         0 |
|     45 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S2       |  -1.44539     |       0.0546113   |   0.992749    |         0 |
|     45 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3a_diff | nan           |     nan           | nan           |         5 |
|     45 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3b_diff |  -0.438244    |       1.06176     |   0.0077975   |         0 |
|     45 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3b_std  |   1.02745     |       2.52745     |   0.0443311   |         0 |
|     45 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S2       |  -4.00262e-05 |       1.49996     |   0.0387102   |         0 |
|     45 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3a_diff |  -0.0715447   |       1.42846     |   0.0730671   |         0 |
|     45 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3b_diff |  -0.0655691   |       1.43443     |   0.0610867   |         0 |
|     45 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3b_std  |  -0.00532641  |       1.49467     |   0.0488746   |         0 |
|     45 | M3_Ellip_SIGReg | test_iso_energy | T3       | S2       |  -2.23898e-06 |       1.5         |   0.000170962 |         0 |
|     45 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3a_diff |  -0.00675609  |       1.49324     |   0.000923286 |         0 |
|     45 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3b_diff |  -0.00589999  |       1.4941      |   0.000728806 |         0 |
|     45 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3b_std  |  -0.000313345 |       1.49969     |   0.000222069 |         0 |
|     45 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S2       |  -0.00012317  |       1.49988     |   0.869416    |         0 |
|     45 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3a_diff |  -0.188407    |       1.31159     |   0.925941    |         0 |
|     45 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3b_diff |  -0.187709    |       1.31229     |   0.917486    |         0 |
|     45 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3b_std  |  -0.0136665   |       1.48633     |   0.875171    |         0 |
|     45 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S2       |   1.8613e-07  |       1.5         |   3.22322e-06 |         0 |
|     45 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3a_diff |   0.0036629   |       1.50366     |   0.0009093   |         0 |
|     45 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3b_diff |   0.00157192  |       1.50157     |   0.000167338 |         0 |
|     45 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3b_std  |   0.000394091 |       1.50039     |   0.00092497  |         0 |
|     46 | M0_Koopman      | test_id_ellip   | T1       | S2       |  -1.45224     |       0.0477579   |   0.991983    |         0 |
|     46 | M0_Koopman      | test_id_ellip   | T1       | S3a_diff |  -1.74705     |       0.247048    |   0.809283    |         0 |
|     46 | M0_Koopman      | test_id_ellip   | T1       | S3b_diff |  -1.49247     |       0.00753351  |   0.999394    |         0 |
|     46 | M0_Koopman      | test_id_ellip   | T1       | S3b_std  |  -1.49961     |       0.000389303 |   0.999593    |         0 |
|     46 | M0_Koopman      | test_id_ellip   | T3       | S2       |   0.000312176 |       1.50031     |   9.53332e-05 |         0 |
|     46 | M0_Koopman      | test_id_ellip   | T3       | S3a_diff |  -0.000325942 |       1.49967     |   9.44549e-05 |         0 |
|     46 | M0_Koopman      | test_id_ellip   | T3       | S3b_diff |  -0.00013466  |       1.49987     |   1.68026e-05 |         0 |
|     46 | M0_Koopman      | test_id_ellip   | T3       | S3b_std  |  -1.5608e-05  |       1.49998     |   2.23847e-07 |         0 |
|     46 | M0_Koopman      | test_ood_ecc    | T1       | S2       |  -1.45602     |       0.0439753   |   0.991783    |         0 |
|     46 | M0_Koopman      | test_ood_ecc    | T1       | S3a_diff |  -1.49176     |       0.00823908  |   0.88131     |         0 |
|     46 | M0_Koopman      | test_ood_ecc    | T1       | S3b_diff |  -1.48471     |       0.0152945   |   0.998885    |         0 |
|     46 | M0_Koopman      | test_ood_ecc    | T1       | S3b_std  |  -1.48971     |       0.010288    |   0.999725    |         0 |
|     46 | M0_Koopman      | test_ood_ecc    | T3       | S2       |  -0.0010812   |       1.49892     |   0.000834509 |         0 |
|     46 | M0_Koopman      | test_ood_ecc    | T3       | S3a_diff |  -0.00202783  |       1.49797     |   0.00283316  |         0 |
|     46 | M0_Koopman      | test_ood_ecc    | T3       | S3b_diff |  -0.00227867  |       1.49772     |   0.00372968  |         0 |
|     46 | M0_Koopman      | test_ood_ecc    | T3       | S3b_std  |  -0.00123474  |       1.49877     |   0.00101759  |         0 |
|     46 | M0_Koopman      | test_iso_energy | T3       | S2       |  -0.00373355  |       1.49627     |   0.0131531   |         0 |
|     46 | M0_Koopman      | test_iso_energy | T3       | S3a_diff |  -0.00375771  |       1.49624     |   0.0128403   |         0 |
|     46 | M0_Koopman      | test_iso_energy | T3       | S3b_diff |  -0.00378371  |       1.49622     |   0.0133554   |         0 |
|     46 | M0_Koopman      | test_iso_energy | T3       | S3b_std  |  -0.00358318  |       1.49642     |   0.0118081   |         0 |
|     46 | M0_Koopman      | test_ellip_peri | T3       | S2       |  -0.0207557   |       1.47924     |   0.843783    |         0 |
|     46 | M0_Koopman      | test_ellip_peri | T3       | S3a_diff |  -0.023015    |       1.47698     |   0.883019    |         0 |
|     46 | M0_Koopman      | test_ellip_peri | T3       | S3b_diff |  -0.0226105   |       1.47739     |   0.884366    |         0 |
|     46 | M0_Koopman      | test_ellip_peri | T3       | S3b_std  |  -0.0219895   |       1.47801     |   0.857724    |         0 |
|     46 | M0_Koopman      | test_ellip_apo  | T3       | S2       |  -0.00350317  |       1.4965      |   0.266496    |         0 |
|     46 | M0_Koopman      | test_ellip_apo  | T3       | S3a_diff |  -0.00471204  |       1.49529     |   0.21984     |         0 |
|     46 | M0_Koopman      | test_ellip_apo  | T3       | S3b_diff |  -0.00454895  |       1.49545     |   0.212072    |         0 |
|     46 | M0_Koopman      | test_ellip_apo  | T3       | S3b_std  |  -0.00377759  |       1.49622     |   0.268588    |         0 |
|     46 | M1_circ         | test_id_ellip   | T1       | S2       |  -1.45195     |       0.0480468   |   0.992035    |         0 |
|     46 | M1_circ         | test_id_ellip   | T1       | S3a_diff |  -1.73916     |       0.239165    |   0.812103    |         0 |
|     46 | M1_circ         | test_id_ellip   | T1       | S3b_diff |  -1.49214     |       0.0078573   |   0.999399    |         0 |
|     46 | M1_circ         | test_id_ellip   | T1       | S3b_std  |  -1.49964     |       0.000364096 |   0.999602    |         0 |
|     46 | M1_circ         | test_id_ellip   | T3       | S2       |  -1.2133      |       0.286703    |   0.739448    |         0 |
|     46 | M1_circ         | test_id_ellip   | T3       | S3a_diff |  -1.39042     |       0.109575    |   0.765114    |         0 |
|     46 | M1_circ         | test_id_ellip   | T3       | S3b_diff |  -1.39045     |       0.109548    |   0.765055    |         0 |
|     46 | M1_circ         | test_id_ellip   | T3       | S3b_std  |  -1.39517     |       0.10483     |   0.765735    |         0 |
|     46 | M1_circ         | test_ood_ecc    | T1       | S2       |  -1.45679     |       0.0432092   |   0.991656    |         0 |
|     46 | M1_circ         | test_ood_ecc    | T1       | S3a_diff |  -1.48651     |       0.0134871   |   0.88265     |         0 |
|     46 | M1_circ         | test_ood_ecc    | T1       | S3b_diff |  -1.48459     |       0.0154085   |   0.998877    |         0 |
|     46 | M1_circ         | test_ood_ecc    | T1       | S3b_std  |  -1.48979     |       0.0102108   |   0.99971     |         0 |
|     46 | M1_circ         | test_ood_ecc    | T3       | S2       |  -0.849123    |       0.650877    |   0.42734     |         0 |
|     46 | M1_circ         | test_ood_ecc    | T3       | S3a_diff |  -1.35662     |       0.143376    |   0.573471    |         0 |
|     46 | M1_circ         | test_ood_ecc    | T3       | S3b_diff |  -1.35716     |       0.142844    |   0.573332    |         0 |
|     46 | M1_circ         | test_ood_ecc    | T3       | S3b_std  |  -1.23665     |       0.26335     |   0.551865    |         0 |
|     46 | M1_circ         | test_iso_energy | T3       | S2       |  -0.94622     |       0.55378     |   0.376219    |         0 |
|     46 | M1_circ         | test_iso_energy | T3       | S3a_diff |  -1.2492      |       0.250797    |   0.438287    |         0 |
|     46 | M1_circ         | test_iso_energy | T3       | S3b_diff |  -1.24926     |       0.250739    |   0.438294    |         0 |
|     46 | M1_circ         | test_iso_energy | T3       | S3b_std  |  -1.21999     |       0.280015    |   0.435481    |         0 |
|     46 | M1_circ         | test_ellip_peri | T3       | S2       |  -0.610247    |       0.889753    |   0.542732    |         0 |
|     46 | M1_circ         | test_ellip_peri | T3       | S3a_diff |  -0.609857    |       0.890143    |   0.542288    |         0 |
|     46 | M1_circ         | test_ellip_peri | T3       | S3b_diff |  -0.610057    |       0.889943    |   0.542713    |         0 |
|     46 | M1_circ         | test_ellip_peri | T3       | S3b_std  |  -0.608994    |       0.891006    |   0.541651    |         0 |
|     46 | M1_circ         | test_ellip_apo  | T3       | S2       |  -0.868808    |       0.631192    |   0.860978    |         0 |
|     46 | M1_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.42463     |       0.0753718   |   0.999778    |         0 |
|     46 | M1_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.42487     |       0.0751321   |   0.999777    |         0 |
|     46 | M1_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.33011     |       0.169893    |   0.99299     |         0 |
|     46 | M2_circ         | test_id_ellip   | T1       | S2       |  -1.45181     |       0.0481876   |   0.992071    |         0 |
|     46 | M2_circ         | test_id_ellip   | T1       | S3a_diff |  -1.74241     |       0.242411    |   0.821075    |         0 |
|     46 | M2_circ         | test_id_ellip   | T1       | S3b_diff |  -1.49231     |       0.00768673  |   0.999445    |         0 |
|     46 | M2_circ         | test_id_ellip   | T1       | S3b_std  |  -1.49975     |       0.000248716 |   0.99962     |         0 |
|     46 | M2_circ         | test_id_ellip   | T3       | S2       |  -1.05309     |       0.44691     |   0.701549    |         0 |
|     46 | M2_circ         | test_id_ellip   | T3       | S3a_diff |  -1.18636     |       0.313643    |   0.722751    |         0 |
|     46 | M2_circ         | test_id_ellip   | T3       | S3b_diff |  -1.18761     |       0.312387    |   0.723351    |         0 |
|     46 | M2_circ         | test_id_ellip   | T3       | S3b_std  |  -1.2237      |       0.276298    |   0.724344    |         0 |
|     46 | M2_circ         | test_ood_ecc    | T1       | S2       |  -1.45778     |       0.0422244   |   0.991509    |         0 |
|     46 | M2_circ         | test_ood_ecc    | T1       | S3a_diff |  -1.52132     |       0.0213189   |   0.880276    |         0 |
|     46 | M2_circ         | test_ood_ecc    | T1       | S3b_diff |  -1.48462     |       0.0153807   |   0.998912    |         0 |
|     46 | M2_circ         | test_ood_ecc    | T1       | S3b_std  |  -1.49068     |       0.0093153   |   0.999707    |         0 |
|     46 | M2_circ         | test_ood_ecc    | T3       | S2       |  -0.759678    |       0.740322    |   0.561326    |         0 |
|     46 | M2_circ         | test_ood_ecc    | T3       | S3a_diff |  -1.16896     |       0.331041    |   0.644015    |         0 |
|     46 | M2_circ         | test_ood_ecc    | T3       | S3b_diff |  -1.177       |       0.323004    |   0.645739    |         0 |
|     46 | M2_circ         | test_ood_ecc    | T3       | S3b_std  |  -1.11827     |       0.381726    |   0.642961    |         0 |
|     46 | M2_circ         | test_iso_energy | T3       | S2       |  -0.892411    |       0.607589    |   0.421865    |         0 |
|     46 | M2_circ         | test_iso_energy | T3       | S3a_diff |  -1.12852     |       0.371481    |   0.469549    |         0 |
|     46 | M2_circ         | test_iso_energy | T3       | S3b_diff |  -1.13335     |       0.366648    |   0.469687    |         0 |
|     46 | M2_circ         | test_iso_energy | T3       | S3b_std  |  -1.13616     |       0.363837    |   0.471372    |         0 |
|     46 | M2_circ         | test_ellip_peri | T3       | S2       |   0.221801    |       1.7218      |   0.096628    |         0 |
|     46 | M2_circ         | test_ellip_peri | T3       | S3a_diff |   0.221013    |       1.72101     |   0.0963288   |         0 |
|     46 | M2_circ         | test_ellip_peri | T3       | S3b_diff |   0.221372    |       1.72137     |   0.0965215   |         0 |
|     46 | M2_circ         | test_ellip_peri | T3       | S3b_std  |   0.222046    |       1.72205     |   0.0969811   |         0 |
|     46 | M2_circ         | test_ellip_apo  | T3       | S2       |  -0.892154    |       0.607846    |   0.863533    |         0 |
|     46 | M2_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.38991     |       0.110091    |   0.994936    |         0 |
|     46 | M2_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.35267     |       0.147331    |   0.995601    |         0 |
|     46 | M2_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.36884     |       0.131164    |   0.983032    |         0 |
|     46 | M3_circ         | test_id_ellip   | T1       | S2       |  -1.46495     |       0.0350499   |   0.736075    |         0 |
|     46 | M3_circ         | test_id_ellip   | T1       | S3a_diff | nan           |     nan           | nan           |         3 |
|     46 | M3_circ         | test_id_ellip   | T1       | S3b_diff |  -1.07204     |       0.427955    |   0.0764219   |         0 |
|     46 | M3_circ         | test_id_ellip   | T1       | S3b_std  |  -1.0906      |       0.409397    |   0.125436    |         0 |
|     46 | M3_circ         | test_id_ellip   | T3       | S2       |  -0.651816    |       0.848184    |   0.126935    |         0 |
|     46 | M3_circ         | test_id_ellip   | T3       | S3a_diff |  -0.835487    |       0.664513    |   0.171069    |         0 |
|     46 | M3_circ         | test_id_ellip   | T3       | S3b_diff |  -0.718729    |       0.781271    |   0.127872    |         0 |
|     46 | M3_circ         | test_id_ellip   | T3       | S3b_std  |  -0.715978    |       0.784022    |   0.100923    |         0 |
|     46 | M3_circ         | test_ood_ecc    | T1       | S2       |  -1.44784     |       0.0521552   |   0.992945    |         0 |
|     46 | M3_circ         | test_ood_ecc    | T1       | S3a_diff | nan           |     nan           | nan           |         8 |
|     46 | M3_circ         | test_ood_ecc    | T1       | S3b_diff |   3.16137     |       4.66137     |   0.312116    |         0 |
|     46 | M3_circ         | test_ood_ecc    | T1       | S3b_std  |   1.94416     |       3.44416     |   0.18378     |         0 |
|     46 | M3_circ         | test_ood_ecc    | T3       | S2       |  -0.249468    |       1.25053     |   0.0205404   |         0 |
|     46 | M3_circ         | test_ood_ecc    | T3       | S3a_diff | nan           |     nan           | nan           |         2 |
|     46 | M3_circ         | test_ood_ecc    | T3       | S3b_diff |  -5.3081      |       3.8081      |   0.0933429   |         0 |
|     46 | M3_circ         | test_ood_ecc    | T3       | S3b_std  |  -0.687074    |       0.812926    |   0.0902715   |         0 |
|     46 | M3_circ         | test_iso_energy | T3       | S2       |  -0.700671    |       0.799329    |   0.112236    |         0 |
|     46 | M3_circ         | test_iso_energy | T3       | S3a_diff | nan           |     nan           | nan           |         1 |
|     46 | M3_circ         | test_iso_energy | T3       | S3b_diff |  -1.01059     |       0.489409    |   0.13002     |         0 |
|     46 | M3_circ         | test_iso_energy | T3       | S3b_std  |  -1.10981     |       0.39019     |   0.0109683   |         0 |
|     46 | M3_circ         | test_ellip_peri | T3       | S2       |   0.208552    |       1.70855     |   0.331572    |         0 |
|     46 | M3_circ         | test_ellip_peri | T3       | S3a_diff |   0.19327     |       1.69327     |   0.31013     |         0 |
|     46 | M3_circ         | test_ellip_peri | T3       | S3b_diff |   0.203249    |       1.70325     |   0.315581    |         0 |
|     46 | M3_circ         | test_ellip_peri | T3       | S3b_std  |   0.204136    |       1.70414     |   0.316676    |         0 |
|     46 | M3_circ         | test_ellip_apo  | T3       | S2       |  -1.32162     |       0.178379    |   0.878938    |         0 |
|     46 | M3_circ         | test_ellip_apo  | T3       | S3a_diff |  -1.52451     |       0.0245074   |   0.852234    |         0 |
|     46 | M3_circ         | test_ellip_apo  | T3       | S3b_diff |  -1.47222     |       0.0277808   |   0.887339    |         0 |
|     46 | M3_circ         | test_ellip_apo  | T3       | S3b_std  |  -1.53038     |       0.0303751   |   0.892083    |         0 |
|     46 | M1_Ellip_Short  | test_id_ellip   | T1       | S2       |  -1.45256     |       0.0474447   |   0.992089    |         0 |
|     46 | M1_Ellip_Short  | test_id_ellip   | T1       | S3a_diff |  -1.75077     |       0.250772    |   0.837697    |         0 |
|     46 | M1_Ellip_Short  | test_id_ellip   | T1       | S3b_diff |  -1.49519     |       0.00481103  |   0.999574    |         0 |
|     46 | M1_Ellip_Short  | test_id_ellip   | T1       | S3b_std  |  -1.50023     |       0.000229483 |   0.999679    |         0 |
|     46 | M1_Ellip_Short  | test_id_ellip   | T3       | S2       |  -1.2372      |       0.262797    |   0.816055    |         0 |
|     46 | M1_Ellip_Short  | test_id_ellip   | T3       | S3a_diff |  -1.36284     |       0.137158    |   0.814919    |         0 |
|     46 | M1_Ellip_Short  | test_id_ellip   | T3       | S3b_diff |  -1.36502     |       0.134983    |   0.814136    |         0 |
|     46 | M1_Ellip_Short  | test_id_ellip   | T3       | S3b_std  |  -1.38281     |       0.11719     |   0.812399    |         0 |
|     46 | M1_Ellip_Short  | test_ood_ecc    | T1       | S2       |  -1.45614     |       0.0438562   |   0.991843    |         0 |
|     46 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3a_diff |  -1.95418     |       0.454184    |   0.860353    |         0 |
|     46 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3b_diff |  -2.13072     |       0.63072     |   0.949564    |         0 |
|     46 | M1_Ellip_Short  | test_ood_ecc    | T1       | S3b_std  |  -1.49536     |       0.0046438   |   0.999843    |         0 |
|     46 | M1_Ellip_Short  | test_ood_ecc    | T3       | S2       |  -1.14883     |       0.351171    |   0.524295    |         0 |
|     46 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3a_diff |  -1.73877     |       0.238768    |   0.639063    |         0 |
|     46 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3b_diff |  -1.74159     |       0.241591    |   0.640188    |         0 |
|     46 | M1_Ellip_Short  | test_ood_ecc    | T3       | S3b_std  |  -1.58946     |       0.0894596   |   0.624052    |         0 |
|     46 | M1_Ellip_Short  | test_iso_energy | T3       | S2       |  -1.19567     |       0.304328    |   0.697907    |         0 |
|     46 | M1_Ellip_Short  | test_iso_energy | T3       | S3a_diff |  -1.48329     |       0.0167132   |   0.721927    |         0 |
|     46 | M1_Ellip_Short  | test_iso_energy | T3       | S3b_diff |  -1.48639     |       0.0136107   |   0.721334    |         0 |
|     46 | M1_Ellip_Short  | test_iso_energy | T3       | S3b_std  |  -1.45345     |       0.0465485   |   0.724934    |         0 |
|     46 | M1_Ellip_Short  | test_ellip_peri | T3       | S2       |  -0.680838    |       0.819162    |   0.994506    |         0 |
|     46 | M1_Ellip_Short  | test_ellip_peri | T3       | S3a_diff |  -0.660055    |       0.839945    |   0.994902    |         0 |
|     46 | M1_Ellip_Short  | test_ellip_peri | T3       | S3b_diff |  -0.670198    |       0.829802    |   0.994708    |         0 |
|     46 | M1_Ellip_Short  | test_ellip_peri | T3       | S3b_std  |  -0.676976    |       0.823024    |   0.994767    |         0 |
|     46 | M1_Ellip_Short  | test_ellip_apo  | T3       | S2       |  -1.5172      |       0.017202    |   0.915189    |         0 |
|     46 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3a_diff |  -1.99538     |       0.495379    |   0.999295    |         0 |
|     46 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3b_diff |  -2.00476     |       0.504758    |   0.999259    |         0 |
|     46 | M1_Ellip_Short  | test_ellip_apo  | T3       | S3b_std  |  -1.93339     |       0.433392    |   0.990771    |         0 |
|     46 | M1_Ellip_Inv    | test_id_ellip   | T1       | S2       |  -1.45071     |       0.0492912   |   0.992412    |         0 |
|     46 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3a_diff |  -1.5622      |       0.0621975   |   0.85308     |         0 |
|     46 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3b_diff |  -1.4938      |       0.00620458  |   0.999318    |         0 |
|     46 | M1_Ellip_Inv    | test_id_ellip   | T1       | S3b_std  |  -1.49913     |       0.000868068 |   0.999683    |         0 |
|     46 | M1_Ellip_Inv    | test_id_ellip   | T3       | S2       |  -1.452       |       0.0480007   |   0.991656    |         0 |
|     46 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3a_diff |  -1.5132      |       0.0131976   |   0.995737    |         0 |
|     46 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3b_diff |  -1.51359     |       0.0135944   |   0.995715    |         0 |
|     46 | M1_Ellip_Inv    | test_id_ellip   | T3       | S3b_std  |  -1.51718     |       0.0171753   |   0.994976    |         0 |
|     46 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S2       |  -1.45889     |       0.041111    |   0.991265    |         0 |
|     46 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3a_diff |  -2.42113     |       0.921129    |   0.862699    |         0 |
|     46 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3b_diff |  -2.18423     |       0.684233    |   0.953477    |         0 |
|     46 | M1_Ellip_Inv    | test_ood_ecc    | T1       | S3b_std  |  -1.66354     |       0.163544    |   0.959534    |         0 |
|     46 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S2       |  -1.45475     |       0.0452462   |   0.970825    |         0 |
|     46 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3a_diff |  -1.56032     |       0.0603232   |   0.972197    |         0 |
|     46 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3b_diff |  -1.5621      |       0.0621045   |   0.972047    |         0 |
|     46 | M1_Ellip_Inv    | test_ood_ecc    | T3       | S3b_std  |  -1.57429     |       0.0742946   |   0.970679    |         0 |
|     46 | M1_Ellip_Inv    | test_iso_energy | T3       | S2       |  -1.3917      |       0.108299    |   0.994476    |         0 |
|     46 | M1_Ellip_Inv    | test_iso_energy | T3       | S3a_diff |  -1.53044     |       0.0304376   |   0.990064    |         0 |
|     46 | M1_Ellip_Inv    | test_iso_energy | T3       | S3b_diff |  -1.53132     |       0.0313246   |   0.989995    |         0 |
|     46 | M1_Ellip_Inv    | test_iso_energy | T3       | S3b_std  |  -1.54551     |       0.0455098   |   0.987616    |         0 |
|     46 | M1_Ellip_Inv    | test_ellip_peri | T3       | S2       |  -1.42907     |       0.0709316   |   0.988923    |         0 |
|     46 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3a_diff |  -1.48115     |       0.0188477   |   0.998612    |         0 |
|     46 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3b_diff |  -1.4814      |       0.0186032   |   0.998616    |         0 |
|     46 | M1_Ellip_Inv    | test_ellip_peri | T3       | S3b_std  |  -1.48365     |       0.0163518   |   0.99887     |         0 |
|     46 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S2       |  -1.45149     |       0.0485057   |   0.992496    |         0 |
|     46 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3a_diff |  -1.52735     |       0.0273519   |   0.999705    |         0 |
|     46 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3b_diff |  -1.52827     |       0.0282699   |   0.999683    |         0 |
|     46 | M1_Ellip_Inv    | test_ellip_apo  | T3       | S3b_std  |  -1.53464     |       0.0346373   |   0.999176    |         0 |
|     46 | M2_Ellip_Inv    | test_id_ellip   | T1       | S2       |  -1.44975     |       0.0502528   |   0.992483    |         0 |
|     46 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3a_diff |  -1.62356     |       0.12356     |   0.830993    |         0 |
|     46 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3b_diff |  -1.49462     |       0.00538099  |   0.999293    |         0 |
|     46 | M2_Ellip_Inv    | test_id_ellip   | T1       | S3b_std  |  -1.49921     |       0.000792191 |   0.999679    |         0 |
|     46 | M2_Ellip_Inv    | test_id_ellip   | T3       | S2       |  -1.44601     |       0.0539878   |   0.991703    |         0 |
|     46 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3a_diff |  -1.52002     |       0.0200208   |   0.996502    |         0 |
|     46 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3b_diff |  -1.52061     |       0.0206076   |   0.996473    |         0 |
|     46 | M2_Ellip_Inv    | test_id_ellip   | T3       | S3b_std  |  -1.52551     |       0.0255079   |   0.995839    |         0 |
|     46 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S2       |  -1.46086     |       0.039142    |   0.990945    |         0 |
|     46 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3a_diff |  -2.45466     |       0.954662    |   0.88242     |         0 |
|     46 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3b_diff |  -2.15611     |       0.656108    |   0.952453    |         0 |
|     46 | M2_Ellip_Inv    | test_ood_ecc    | T1       | S3b_std  |  -1.57155     |       0.0715468   |   0.965157    |         0 |
|     46 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S2       |  -1.42883     |       0.0711723   |   0.964237    |         0 |
|     46 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3a_diff |  -1.49143     |       0.00857082  |   0.968873    |         0 |
|     46 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3b_diff |  -1.49292     |       0.00708476  |   0.968851    |         0 |
|     46 | M2_Ellip_Inv    | test_ood_ecc    | T3       | S3b_std  |  -1.50296     |       0.00295811  |   0.968583    |         0 |
|     46 | M2_Ellip_Inv    | test_iso_energy | T3       | S2       |  -1.3876      |       0.112401    |   0.994738    |         0 |
|     46 | M2_Ellip_Inv    | test_iso_energy | T3       | S3a_diff |  -1.52087     |       0.0208735   |   0.993423    |         0 |
|     46 | M2_Ellip_Inv    | test_iso_energy | T3       | S3b_diff |  -1.52194     |       0.021939    |   0.993385    |         0 |
|     46 | M2_Ellip_Inv    | test_iso_energy | T3       | S3b_std  |  -1.53617     |       0.036174    |   0.991919    |         0 |
|     46 | M2_Ellip_Inv    | test_ellip_peri | T3       | S2       |  -1.4364      |       0.0635978   |   0.988298    |         0 |
|     46 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3a_diff |  -1.48143     |       0.0185654   |   0.998052    |         0 |
|     46 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3b_diff |  -1.48163     |       0.0183688   |   0.998059    |         0 |
|     46 | M2_Ellip_Inv    | test_ellip_peri | T3       | S3b_std  |  -1.48286     |       0.0171391   |   0.998382    |         0 |
|     46 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S2       |  -1.44184     |       0.0581556   |   0.992079    |         0 |
|     46 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3a_diff |  -1.5081      |       0.00809823  |   0.999544    |         0 |
|     46 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3b_diff |  -1.50911     |       0.00910843  |   0.99953     |         0 |
|     46 | M2_Ellip_Inv    | test_ellip_apo  | T3       | S3b_std  |  -1.51455     |       0.0145545   |   0.999048    |         0 |
|     46 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S2       |  -1.4298      |       0.0701999   |   0.666494    |         0 |
|     46 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3a_diff | nan           |     nan           | nan           |         8 |
|     46 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3b_diff |  -0.704253    |       0.795747    |   0.0189469   |         0 |
|     46 | M3_Ellip_SIGReg | test_id_ellip   | T1       | S3b_std  |  -0.0879465   |       1.41205     |   0.000292798 |         0 |
|     46 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S2       |   3.79516e-05 |       1.50004     |   0.0939676   |         0 |
|     46 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3a_diff |   0.0609403   |       1.56094     |   0.103282    |         0 |
|     46 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3b_diff |   0.0605393   |       1.56054     |   0.102612    |         0 |
|     46 | M3_Ellip_SIGReg | test_id_ellip   | T3       | S3b_std  |   0.0042257   |       1.50423     |   0.0895246   |         0 |
|     46 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S2       |  -1.45195     |       0.0480544   |   0.99233     |         0 |
|     46 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3a_diff | nan           |     nan           | nan           |        14 |
|     46 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3b_diff |  -2.21652     |       0.71652     |   0.396874    |         0 |
|     46 | M3_Ellip_SIGReg | test_ood_ecc    | T1       | S3b_std  |   0.638797    |       2.1388      |   0.0222859   |         0 |
|     46 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S2       |   1.15188e-05 |       1.50001     |   0.00926539  |         0 |
|     46 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3a_diff |   0.0169008   |       1.5169      |   0.00763292  |         0 |
|     46 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3b_diff |   0.0189389   |       1.51894     |   0.00973592  |         0 |
|     46 | M3_Ellip_SIGReg | test_ood_ecc    | T3       | S3b_std  |   0.00104599  |       1.50105     |   0.00624066  |         0 |
|     46 | M3_Ellip_SIGReg | test_iso_energy | T3       | S2       |   6.43777e-05 |       1.50006     |   0.193569    |         0 |
|     46 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3a_diff |   0.0994426   |       1.59944     |   0.188789    |         0 |
|     46 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3b_diff |   0.0990039   |       1.599       |   0.189438    |         0 |
|     46 | M3_Ellip_SIGReg | test_iso_energy | T3       | S3b_std  |   0.0073745   |       1.50737     |   0.19745     |         0 |
|     46 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S2       |   6.24337e-05 |       1.50006     |   0.92304     |         0 |
|     46 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3a_diff |   0.091354    |       1.59135     |   0.920421    |         0 |
|     46 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3b_diff |   0.0894663   |       1.58947     |   0.92165     |         0 |
|     46 | M3_Ellip_SIGReg | test_ellip_peri | T3       | S3b_std  |   0.00773633  |       1.50774     |   0.928922    |         0 |
|     46 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S2       |   3.88554e-05 |       1.50004     |   0.172755    |         0 |
|     46 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3a_diff |   0.0507665   |       1.55077     |   0.124929    |         0 |
|     46 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3b_diff |   0.0531538   |       1.55315     |   0.143162    |         0 |
|     46 | M3_Ellip_SIGReg | test_ellip_apo  | T3       | S3b_std  |   0.00456846  |       1.50457     |   0.163682    |         0 |

## Pericenter versus apocenter

|   seed | model           |   mean_phase_gap |   invalid |
|--------|-----------------|------------------|-----------|
|     42 | M0_Koopman      |        0.0121674 |         0 |
|     42 | M1_circ         |        1.44896   |         0 |
|     42 | M2_circ         |        1.23787   |         0 |
|     42 | M3_circ         |        0.899705  |         0 |
|     42 | M1_Ellip_Short  |        0.654582  |         0 |
|     42 | M1_Ellip_Inv    |        0.0698113 |         0 |
|     42 | M2_Ellip_Inv    |        0.0537944 |         0 |
|     42 | M3_Ellip_SIGReg |        5.40165   |         0 |
|     43 | M0_Koopman      |        0.0119279 |         0 |
|     43 | M1_circ         |        1.37625   |         0 |
|     43 | M2_circ         |        1.27633   |         0 |
|     43 | M3_circ         |        0.984678  |         0 |
|     43 | M1_Ellip_Short  |        0.586566  |         0 |
|     43 | M1_Ellip_Inv    |        0.0542272 |         0 |
|     43 | M2_Ellip_Inv    |        0.0354331 |         0 |
|     43 | M3_Ellip_SIGReg |        0.165624  |         0 |
|     44 | M0_Koopman      |        0.0206609 |         0 |
|     44 | M1_circ         |        1.55403   |         0 |
|     44 | M2_circ         |        1.24699   |         0 |
|     44 | M3_circ         |        1.26036   |         0 |
|     44 | M1_Ellip_Short  |        0.467749  |         0 |
|     44 | M1_Ellip_Inv    |        0.0504682 |         0 |
|     44 | M2_Ellip_Inv    |        0.0644175 |         0 |
|     44 | M3_Ellip_SIGReg |        0.101585  |         0 |
|     45 | M0_Koopman      |        0.0210689 |         0 |
|     45 | M1_circ         |        1.53549   |         0 |
|     45 | M2_circ         |        1.38756   |         0 |
|     45 | M3_circ         |        0.445508  |         0 |
|     45 | M1_Ellip_Short  |        0.422561  |         0 |
|     45 | M1_Ellip_Inv    |        0.0653437 |         0 |
|     45 | M2_Ellip_Inv    |        0.0255403 |         0 |
|     45 | M3_Ellip_SIGReg |        0.200709  |         0 |
|     46 | M0_Koopman      |        0.0199876 |         0 |
|     46 | M1_circ         |        1.47343   |         0 |
|     46 | M2_circ         |        1.095     |         0 |
|     46 | M3_circ         |        1.21345   |         0 |
|     46 | M1_Ellip_Short  |        0.525079  |         0 |
|     46 | M1_Ellip_Inv    |        0.0395055 |         0 |
|     46 | M2_Ellip_Inv    |        0.03084   |         0 |
|     46 | M3_Ellip_SIGReg |        0.0503036 |         0 |

Per-orbit ratios and absolute log gaps are in the JSON; invalid orbits make the aggregate undefined.

## Isoenergetic dispersion

|   seed | model           |   a |   cv_log_omega |   invalid |
|--------|-----------------|-----|----------------|-----------|
|     42 | M0_Koopman      | 0.7 |     0.0130463  |         0 |
|     42 | M0_Koopman      | 1.2 |     0.0117916  |         0 |
|     42 | M0_Koopman      | 1.8 |     0.0144001  |         0 |
|     42 | M1_circ         | 0.7 |     0.385602   |         0 |
|     42 | M1_circ         | 1.2 |     0.481526   |         0 |
|     42 | M1_circ         | 1.8 |     0.517826   |         0 |
|     42 | M2_circ         | 0.7 |     0.391191   |         0 |
|     42 | M2_circ         | 1.2 |     0.459102   |         0 |
|     42 | M2_circ         | 1.8 |     0.503377   |         0 |
|     42 | M3_circ         | 0.7 |     0.387139   |         0 |
|     42 | M3_circ         | 1.2 |     0.786013   |         0 |
|     42 | M3_circ         | 1.8 |     0.909215   |         0 |
|     42 | M1_Ellip_Short  | 0.7 |     0.278921   |         0 |
|     42 | M1_Ellip_Short  | 1.2 |     0.227388   |         0 |
|     42 | M1_Ellip_Short  | 1.8 |     0.489704   |         0 |
|     42 | M1_Ellip_Inv    | 0.7 |     0.0153981  |         0 |
|     42 | M1_Ellip_Inv    | 1.2 |     0.0524862  |         0 |
|     42 | M1_Ellip_Inv    | 1.8 |     0.105816   |         0 |
|     42 | M2_Ellip_Inv    | 0.7 |     0.0278     |         0 |
|     42 | M2_Ellip_Inv    | 1.2 |     0.0385219  |         0 |
|     42 | M2_Ellip_Inv    | 1.8 |     0.101366   |         0 |
|     42 | M3_Ellip_SIGReg | 0.7 |     0.75335    |         0 |
|     42 | M3_Ellip_SIGReg | 1.2 |     9.13228    |         0 |
|     42 | M3_Ellip_SIGReg | 1.8 |    11.1167     |         0 |
|     43 | M0_Koopman      | 0.7 |     0.00727866 |         0 |
|     43 | M0_Koopman      | 1.2 |     0.00911467 |         0 |
|     43 | M0_Koopman      | 1.8 |     0.00987505 |         0 |
|     43 | M1_circ         | 0.7 |     0.45635    |         0 |
|     43 | M1_circ         | 1.2 |     0.636233   |         0 |
|     43 | M1_circ         | 1.8 |     0.491977   |         0 |
|     43 | M2_circ         | 0.7 |     0.423062   |         0 |
|     43 | M2_circ         | 1.2 |     0.653012   |         0 |
|     43 | M2_circ         | 1.8 |     0.508419   |         0 |
|     43 | M3_circ         | 0.7 |     0.785544   |         0 |
|     43 | M3_circ         | 1.2 |     1.09156    |         0 |
|     43 | M3_circ         | 1.8 |     9.23778    |         0 |
|     43 | M1_Ellip_Short  | 0.7 |     0.290016   |         0 |
|     43 | M1_Ellip_Short  | 1.2 |     0.379789   |         0 |
|     43 | M1_Ellip_Short  | 1.8 |     0.43126    |         0 |
|     43 | M1_Ellip_Inv    | 0.7 |     0.0359556  |         0 |
|     43 | M1_Ellip_Inv    | 1.2 |     0.0776702  |         0 |
|     43 | M1_Ellip_Inv    | 1.8 |     0.0620433  |         0 |
|     43 | M2_Ellip_Inv    | 0.7 |     0.0143293  |         0 |
|     43 | M2_Ellip_Inv    | 1.2 |     0.036171   |         0 |
|     43 | M2_Ellip_Inv    | 1.8 |     0.0616655  |         0 |
|     43 | M3_Ellip_SIGReg | 0.7 |     0.169978   |         0 |
|     43 | M3_Ellip_SIGReg | 1.2 |     0.105566   |         0 |
|     43 | M3_Ellip_SIGReg | 1.8 |     0.134412   |         0 |
|     44 | M0_Koopman      | 0.7 |     0.0143615  |         0 |
|     44 | M0_Koopman      | 1.2 |     0.0118618  |         0 |
|     44 | M0_Koopman      | 1.8 |     0.0101229  |         0 |
|     44 | M1_circ         | 0.7 |     0.471108   |         0 |
|     44 | M1_circ         | 1.2 |     0.479633   |         0 |
|     44 | M1_circ         | 1.8 |     0.555785   |         0 |
|     44 | M2_circ         | 0.7 |     0.373582   |         0 |
|     44 | M2_circ         | 1.2 |     0.439489   |         0 |
|     44 | M2_circ         | 1.8 |     0.486259   |         0 |
|     44 | M3_circ         | 0.7 |     0.706758   |         0 |
|     44 | M3_circ         | 1.2 |     7.73132    |         0 |
|     44 | M3_circ         | 1.8 |     1.08845    |         0 |
|     44 | M1_Ellip_Short  | 0.7 |     0.358643   |         0 |
|     44 | M1_Ellip_Short  | 1.2 |     0.368252   |         0 |
|     44 | M1_Ellip_Short  | 1.8 |     0.50731    |         0 |
|     44 | M1_Ellip_Inv    | 0.7 |     0.0456315  |         0 |
|     44 | M1_Ellip_Inv    | 1.2 |     0.0551736  |         0 |
|     44 | M1_Ellip_Inv    | 1.8 |     0.102775   |         0 |
|     44 | M2_Ellip_Inv    | 0.7 |     0.0208292  |         0 |
|     44 | M2_Ellip_Inv    | 1.2 |     0.0425847  |         0 |
|     44 | M2_Ellip_Inv    | 1.8 |     0.0829863  |         0 |
|     44 | M3_Ellip_SIGReg | 0.7 |     0.129247   |         0 |
|     44 | M3_Ellip_SIGReg | 1.2 |     0.0834587  |         0 |
|     44 | M3_Ellip_SIGReg | 1.8 |     0.0743799  |         0 |
|     45 | M0_Koopman      | 0.7 |     0.0199369  |         0 |
|     45 | M0_Koopman      | 1.2 |     0.0110767  |         0 |
|     45 | M0_Koopman      | 1.8 |     0.0107854  |         0 |
|     45 | M1_circ         | 0.7 |     0.371157   |         0 |
|     45 | M1_circ         | 1.2 |     0.418917   |         0 |
|     45 | M1_circ         | 1.8 |     0.314096   |         0 |
|     45 | M2_circ         | 0.7 |     0.406351   |         0 |
|     45 | M2_circ         | 1.2 |     0.413271   |         0 |
|     45 | M2_circ         | 1.8 |     0.255352   |         0 |
|     45 | M3_circ         | 0.7 |     0.438518   |         0 |
|     45 | M3_circ         | 1.2 |     1.39598    |         0 |
|     45 | M3_circ         | 1.8 |     0.827493   |         0 |
|     45 | M1_Ellip_Short  | 0.7 |     0.340667   |         0 |
|     45 | M1_Ellip_Short  | 1.2 |     0.296253   |         0 |
|     45 | M1_Ellip_Short  | 1.8 |     0.299683   |         0 |
|     45 | M1_Ellip_Inv    | 0.7 |     0.0489861  |         0 |
|     45 | M1_Ellip_Inv    | 1.2 |     0.0450649  |         0 |
|     45 | M1_Ellip_Inv    | 1.8 |     0.0617042  |         0 |
|     45 | M2_Ellip_Inv    | 0.7 |     0.0740094  |         0 |
|     45 | M2_Ellip_Inv    | 1.2 |     0.0391778  |         0 |
|     45 | M2_Ellip_Inv    | 1.8 |     0.0738039  |         0 |
|     45 | M3_Ellip_SIGReg | 0.7 |     0.081122   |         0 |
|     45 | M3_Ellip_SIGReg | 1.2 |     0.0803544  |         0 |
|     45 | M3_Ellip_SIGReg | 1.8 |     0.0915257  |         0 |
|     46 | M0_Koopman      | 0.7 |     0.0107191  |         0 |
|     46 | M0_Koopman      | 1.2 |     0.0118892  |         0 |
|     46 | M0_Koopman      | 1.8 |     0.014309   |         0 |
|     46 | M1_circ         | 0.7 |     0.408591   |         0 |
|     46 | M1_circ         | 1.2 |     0.453921   |         0 |
|     46 | M1_circ         | 1.8 |     0.719115   |         0 |
|     46 | M2_circ         | 0.7 |     0.312841   |         0 |
|     46 | M2_circ         | 1.2 |     0.432284   |         0 |
|     46 | M2_circ         | 1.8 |     0.603625   |         0 |
|     46 | M3_circ         | 0.7 |     0.741651   |         0 |
|     46 | M3_circ         | 1.2 |     0.757663   |         0 |
|     46 | M3_circ         | 1.8 |     1.38118    |         0 |
|     46 | M1_Ellip_Short  | 0.7 |     0.328121   |         0 |
|     46 | M1_Ellip_Short  | 1.2 |     0.410193   |         0 |
|     46 | M1_Ellip_Short  | 1.8 |     0.327546   |         0 |
|     46 | M1_Ellip_Inv    | 0.7 |     0.0330264  |         0 |
|     46 | M1_Ellip_Inv    | 1.2 |     0.0445343  |         0 |
|     46 | M1_Ellip_Inv    | 1.8 |     0.0864746  |         0 |
|     46 | M2_Ellip_Inv    | 0.7 |     0.0285071  |         0 |
|     46 | M2_Ellip_Inv    | 1.2 |     0.0457431  |         0 |
|     46 | M2_Ellip_Inv    | 1.8 |     0.0627803  |         0 |
|     46 | M3_Ellip_SIGReg | 0.7 |     0.0720873  |         0 |
|     46 | M3_Ellip_SIGReg | 1.2 |     0.0904107  |         0 |
|     46 | M3_Ellip_SIGReg | 1.8 |     0.0729698  |         0 |

cv_log_omega is the population standard deviation of log frequency, as specified; it is not std/mean.

## Train-only physical probes

|   seed | model           | split         | feature   | method   | target         |               r2 |
|--------|-----------------|---------------|-----------|----------|----------------|------------------|
|     42 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_r0         |     -0.0754351   |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_r0         |      0.18907     |
|     42 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_v0         |     -0.0748765   |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.187998    |
|     42 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_a          |      0.132195    |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_a          |      0.406506    |
|     42 | M0_Koopman      | test_id_ellip | norm      | ridge    | e              |      0.000897216 |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | e              |   -117.003       |
|     42 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_h          |     -0.0179112   |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_h          |     -4.50257     |
|     42 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_abs_energy |      0.132195    |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.406506    |
|     42 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_r0         |     -0.0969915   |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_r0         |      0.176652    |
|     42 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_v0         |     -0.00584858  |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.170569    |
|     42 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_a          |      0.0534986   |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_a          |      0.272256    |
|     42 | M0_Koopman      | test_id_ellip | norm      | poly2    | e              |     -0.0194436   |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | e              |   -116.655       |
|     42 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_h          |     -0.112146    |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_h          |     -4.54636     |
|     42 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_abs_energy |      0.0534986   |
|     42 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.272256    |
|     42 | M0_Koopman      | test_id_ellip | state     | ridge    | log_r0         |     -0.255927    |
|     42 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_r0         |      0.0406406   |
|     42 | M0_Koopman      | test_id_ellip | state     | ridge    | log_v0         |      0.0113393   |
|     42 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_v0         |     -0.113215    |
|     42 | M0_Koopman      | test_id_ellip | state     | ridge    | log_a          |     -0.731863    |
|     42 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_a          |     -1.80071     |
|     42 | M0_Koopman      | test_id_ellip | state     | ridge    | e              |     -4.22716     |
|     42 | M0_Koopman      | test_ood_ecc  | state     | ridge    | e              |   -111.689       |
|     42 | M0_Koopman      | test_id_ellip | state     | ridge    | log_h          |     -1.21616     |
|     42 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_h          |     -7.21506     |
|     42 | M0_Koopman      | test_id_ellip | state     | ridge    | log_abs_energy |     -0.731863    |
|     42 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.80071     |
|     42 | M0_Koopman      | test_id_ellip | state     | poly2    | log_r0         |      0.862465    |
|     42 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_r0         |      0.960652    |
|     42 | M0_Koopman      | test_id_ellip | state     | poly2    | log_v0         |     -2.0858      |
|     42 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_v0         |      0.402127    |
|     42 | M0_Koopman      | test_id_ellip | state     | poly2    | log_a          |     -4.81347     |
|     42 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_a          |     -0.28994     |
|     42 | M0_Koopman      | test_id_ellip | state     | poly2    | e              |    -16.1318      |
|     42 | M0_Koopman      | test_ood_ecc  | state     | poly2    | e              |   -677.266       |
|     42 | M0_Koopman      | test_id_ellip | state     | poly2    | log_h          |      0.898918    |
|     42 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_h          |     -0.588625    |
|     42 | M0_Koopman      | test_id_ellip | state     | poly2    | log_abs_energy |     -4.81347     |
|     42 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.28994     |
|     42 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_r0         |      0.930556    |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.836668    |
|     42 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_v0         |      0.756261    |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.541772    |
|     42 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_a          |      0.573196    |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_a          |      0.272827    |
|     42 | M0_Koopman      | test_id_ellip | log_omega | ridge    | e              |     -0.347314    |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | e              |    -75.1981      |
|     42 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_h          |      0.186219    |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_h          |     -1.95419     |
|     42 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.573196    |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.272827    |
|     42 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_r0         |      0.917361    |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.962025    |
|     42 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_v0         |      0.0947049   |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.733151    |
|     42 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_a          |     -3.74585     |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_a          |     -0.937698    |
|     42 | M0_Koopman      | test_id_ellip | log_omega | poly2    | e              |    -58.8965      |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | e              |   -401.012       |
|     42 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_h          |     -8.26149     |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_h          |     -2.28149     |
|     42 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_abs_energy |     -3.74585     |
|     42 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -0.937698    |
|     42 | M1_circ         | test_id_ellip | norm      | ridge    | log_r0         |     -0.0239699   |
|     42 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |      0.204427    |
|     42 | M1_circ         | test_id_ellip | norm      | ridge    | log_v0         |     -0.0386807   |
|     42 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.17343     |
|     42 | M1_circ         | test_id_ellip | norm      | ridge    | log_a          |      0.170359    |
|     42 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_a          |      0.392008    |
|     42 | M1_circ         | test_id_ellip | norm      | ridge    | e              |      0.000847955 |
|     42 | M1_circ         | test_ood_ecc  | norm      | ridge    | e              |   -116.961       |
|     42 | M1_circ         | test_id_ellip | norm      | ridge    | log_h          |      0.0118761   |
|     42 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -4.56955     |
|     42 | M1_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |      0.170359    |
|     42 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.392008    |
|     42 | M1_circ         | test_id_ellip | norm      | poly2    | log_r0         |     -0.0445604   |
|     42 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |      0.191044    |
|     42 | M1_circ         | test_id_ellip | norm      | poly2    | log_v0         |      0.0223904   |
|     42 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.159017    |
|     42 | M1_circ         | test_id_ellip | norm      | poly2    | log_a          |      0.103284    |
|     42 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_a          |      0.243155    |
|     42 | M1_circ         | test_id_ellip | norm      | poly2    | e              |     -0.0219227   |
|     42 | M1_circ         | test_ood_ecc  | norm      | poly2    | e              |   -116.88        |
|     42 | M1_circ         | test_id_ellip | norm      | poly2    | log_h          |     -0.0758726   |
|     42 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -4.66175     |
|     42 | M1_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |      0.103284    |
|     42 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.243155    |
|     42 | M1_circ         | test_id_ellip | state     | ridge    | log_r0         |     -0.606851    |
|     42 | M1_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -0.108742    |
|     42 | M1_circ         | test_id_ellip | state     | ridge    | log_v0         |     -0.383895    |
|     42 | M1_circ         | test_ood_ecc  | state     | ridge    | log_v0         |     -0.124123    |
|     42 | M1_circ         | test_id_ellip | state     | ridge    | log_a          |     -1.03456     |
|     42 | M1_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -2.14061     |
|     42 | M1_circ         | test_id_ellip | state     | ridge    | e              |     -2.60099     |
|     42 | M1_circ         | test_ood_ecc  | state     | ridge    | e              |   -136.783       |
|     42 | M1_circ         | test_id_ellip | state     | ridge    | log_h          |     -0.426774    |
|     42 | M1_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -7.38263     |
|     42 | M1_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -1.03456     |
|     42 | M1_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.14061     |
|     42 | M1_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.92677     |
|     42 | M1_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.996701    |
|     42 | M1_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.443395    |
|     42 | M1_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.894617    |
|     42 | M1_circ         | test_id_ellip | state     | poly2    | log_a          |     -7.86439     |
|     42 | M1_circ         | test_ood_ecc  | state     | poly2    | log_a          |      0.659993    |
|     42 | M1_circ         | test_id_ellip | state     | poly2    | e              |    -27.2749      |
|     42 | M1_circ         | test_ood_ecc  | state     | poly2    | e              |   -253.934       |
|     42 | M1_circ         | test_id_ellip | state     | poly2    | log_h          |      0.855642    |
|     42 | M1_circ         | test_ood_ecc  | state     | poly2    | log_h          |      0.693013    |
|     42 | M1_circ         | test_id_ellip | state     | poly2    | log_abs_energy |     -7.86439     |
|     42 | M1_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.659993    |
|     42 | M1_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.931188    |
|     42 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.899688    |
|     42 | M1_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.981541    |
|     42 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.932517    |
|     42 | M1_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.581308    |
|     42 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |      0.391982    |
|     42 | M1_circ         | test_id_ellip | log_omega | ridge    | e              |      0.523629    |
|     42 | M1_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -26.0216      |
|     42 | M1_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.569211    |
|     42 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -0.781546    |
|     42 | M1_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.581308    |
|     42 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.391982    |
|     42 | M1_circ         | test_id_ellip | log_omega | poly2    | log_r0         |      0.936967    |
|     42 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.770767    |
|     42 | M1_circ         | test_id_ellip | log_omega | poly2    | log_v0         |      0.98834     |
|     42 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.957269    |
|     42 | M1_circ         | test_id_ellip | log_omega | poly2    | log_a          |     -0.0356079   |
|     42 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |     -0.439027    |
|     42 | M1_circ         | test_id_ellip | log_omega | poly2    | e              |      0.391084    |
|     42 | M1_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -30.3593      |
|     42 | M1_circ         | test_id_ellip | log_omega | poly2    | log_h          |      0.406082    |
|     42 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.652625    |
|     42 | M1_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |     -0.0356079   |
|     42 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -0.439027    |
|     42 | M2_circ         | test_id_ellip | norm      | ridge    | log_r0         |     -0.0471824   |
|     42 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |      0.201086    |
|     42 | M2_circ         | test_id_ellip | norm      | ridge    | log_v0         |     -0.0478195   |
|     42 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.151696    |
|     42 | M2_circ         | test_id_ellip | norm      | ridge    | log_a          |      0.133697    |
|     42 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_a          |      0.30456     |
|     42 | M2_circ         | test_id_ellip | norm      | ridge    | e              |      0.00345774  |
|     42 | M2_circ         | test_ood_ecc  | norm      | ridge    | e              |   -117.274       |
|     42 | M2_circ         | test_id_ellip | norm      | ridge    | log_h          |     -0.0274423   |
|     42 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -4.88713     |
|     42 | M2_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |      0.133697    |
|     42 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.30456     |
|     42 | M2_circ         | test_id_ellip | norm      | poly2    | log_r0         |     -0.0647267   |
|     42 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |      0.191734    |
|     42 | M2_circ         | test_id_ellip | norm      | poly2    | log_v0         |      0.0166705   |
|     42 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.145348    |
|     42 | M2_circ         | test_id_ellip | norm      | poly2    | log_a          |      0.0623529   |
|     42 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_a          |      0.157628    |
|     42 | M2_circ         | test_id_ellip | norm      | poly2    | e              |     -0.0182289   |
|     42 | M2_circ         | test_ood_ecc  | norm      | poly2    | e              |   -117.922       |
|     42 | M2_circ         | test_id_ellip | norm      | poly2    | log_h          |     -0.119406    |
|     42 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -5.08301     |
|     42 | M2_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |      0.0623529   |
|     42 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.157628    |
|     42 | M2_circ         | test_id_ellip | state     | ridge    | log_r0         |     -0.421275    |
|     42 | M2_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -0.924655    |
|     42 | M2_circ         | test_id_ellip | state     | ridge    | log_v0         |     -0.353744    |
|     42 | M2_circ         | test_ood_ecc  | state     | ridge    | log_v0         |     -0.514307    |
|     42 | M2_circ         | test_id_ellip | state     | ridge    | log_a          |     -0.716815    |
|     42 | M2_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -2.08248     |
|     42 | M2_circ         | test_id_ellip | state     | ridge    | e              |     -3.82579     |
|     42 | M2_circ         | test_ood_ecc  | state     | ridge    | e              |   -149.515       |
|     42 | M2_circ         | test_id_ellip | state     | ridge    | log_h          |     -0.790479    |
|     42 | M2_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -9.39078     |
|     42 | M2_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -0.716815    |
|     42 | M2_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.08248     |
|     42 | M2_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.930513    |
|     42 | M2_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.964248    |
|     42 | M2_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.960627    |
|     42 | M2_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.924279    |
|     42 | M2_circ         | test_id_ellip | state     | poly2    | log_a          |      0.115037    |
|     42 | M2_circ         | test_ood_ecc  | state     | poly2    | log_a          |     -0.748343    |
|     42 | M2_circ         | test_id_ellip | state     | poly2    | e              |     -4.48504     |
|     42 | M2_circ         | test_ood_ecc  | state     | poly2    | e              |   -311.244       |
|     42 | M2_circ         | test_id_ellip | state     | poly2    | log_h          |      0.339035    |
|     42 | M2_circ         | test_ood_ecc  | state     | poly2    | log_h          |      0.753262    |
|     42 | M2_circ         | test_id_ellip | state     | poly2    | log_abs_energy |      0.115037    |
|     42 | M2_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.748343    |
|     42 | M2_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.884386    |
|     42 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.868429    |
|     42 | M2_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.944507    |
|     42 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.801106    |
|     42 | M2_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.727501    |
|     42 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |      0.52681     |
|     42 | M2_circ         | test_id_ellip | log_omega | ridge    | e              |      0.178706    |
|     42 | M2_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -69.6724      |
|     42 | M2_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.589573    |
|     42 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -1.42394     |
|     42 | M2_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.727501    |
|     42 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.52681     |
|     42 | M2_circ         | test_id_ellip | log_omega | poly2    | log_r0         |      0.932466    |
|     42 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.891065    |
|     42 | M2_circ         | test_id_ellip | log_omega | poly2    | log_v0         |      0.94039     |
|     42 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.86548     |
|     42 | M2_circ         | test_id_ellip | log_omega | poly2    | log_a          |      0.892655    |
|     42 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |      0.593262    |
|     42 | M2_circ         | test_id_ellip | log_omega | poly2    | e              |      0.403124    |
|     42 | M2_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -39.6422      |
|     42 | M2_circ         | test_id_ellip | log_omega | poly2    | log_h          |      0.92623     |
|     42 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.0895191   |
|     42 | M2_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.892655    |
|     42 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.593262    |
|     42 | M3_circ         | test_id_ellip | norm      | ridge    | log_r0         |     -0.834804    |
|     42 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.127603    |
|     42 | M3_circ         | test_id_ellip | norm      | ridge    | log_v0         |     -0.349408    |
|     42 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.0949922   |
|     42 | M3_circ         | test_id_ellip | norm      | ridge    | log_a          |     -1.33482     |
|     42 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_a          |     -1.33482     |
|     42 | M3_circ         | test_id_ellip | norm      | ridge    | e              |     -0.002616    |
|     42 | M3_circ         | test_ood_ecc  | norm      | ridge    | e              |   -116.28        |
|     42 | M3_circ         | test_id_ellip | norm      | ridge    | log_h          |     -1.07162     |
|     42 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -7.95375     |
|     42 | M3_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |     -1.33482     |
|     42 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -1.33482     |
|     42 | M3_circ         | test_id_ellip | norm      | poly2    | log_r0         |     -0.834804    |
|     42 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.127603    |
|     42 | M3_circ         | test_id_ellip | norm      | poly2    | log_v0         |     -0.349408    |
|     42 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0949922   |
|     42 | M3_circ         | test_id_ellip | norm      | poly2    | log_a          |     -1.33482     |
|     42 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_a          |     -1.33482     |
|     42 | M3_circ         | test_id_ellip | norm      | poly2    | e              |     -0.002616    |
|     42 | M3_circ         | test_ood_ecc  | norm      | poly2    | e              |   -116.28        |
|     42 | M3_circ         | test_id_ellip | norm      | poly2    | log_h          |     -1.07162     |
|     42 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -7.95375     |
|     42 | M3_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |     -1.33482     |
|     42 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.33482     |
|     42 | M3_circ         | test_id_ellip | state     | ridge    | log_r0         |     -1.41254     |
|     42 | M3_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -0.958125    |
|     42 | M3_circ         | test_id_ellip | state     | ridge    | log_v0         |     -0.731113    |
|     42 | M3_circ         | test_ood_ecc  | state     | ridge    | log_v0         |     -0.359076    |
|     42 | M3_circ         | test_id_ellip | state     | ridge    | log_a          |     -2.13805     |
|     42 | M3_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -3.19155     |
|     42 | M3_circ         | test_id_ellip | state     | ridge    | e              |     -0.686764    |
|     42 | M3_circ         | test_ood_ecc  | state     | ridge    | e              |    -98.961       |
|     42 | M3_circ         | test_id_ellip | state     | ridge    | log_h          |     -1.46452     |
|     42 | M3_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -8.91915     |
|     42 | M3_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -2.13805     |
|     42 | M3_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -3.19155     |
|     42 | M3_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.668423    |
|     42 | M3_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.567914    |
|     42 | M3_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.58025     |
|     42 | M3_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.584353    |
|     42 | M3_circ         | test_id_ellip | state     | poly2    | log_a          |      0.159015    |
|     42 | M3_circ         | test_ood_ecc  | state     | poly2    | log_a          |     -0.499493    |
|     42 | M3_circ         | test_id_ellip | state     | poly2    | e              |     -1.25095     |
|     42 | M3_circ         | test_ood_ecc  | state     | poly2    | e              |    -76.8411      |
|     42 | M3_circ         | test_id_ellip | state     | poly2    | log_h          |     -0.285078    |
|     42 | M3_circ         | test_ood_ecc  | state     | poly2    | log_h          |     -3.10318     |
|     42 | M3_circ         | test_id_ellip | state     | poly2    | log_abs_energy |      0.159015    |
|     42 | M3_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.499493    |
|     42 | M3_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.631038    |
|     42 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.469383    |
|     42 | M3_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.611685    |
|     42 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.167764    |
|     42 | M3_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.252249    |
|     42 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |     -0.778719    |
|     42 | M3_circ         | test_id_ellip | log_omega | ridge    | e              |      0.0110533   |
|     42 | M3_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -91.0113      |
|     42 | M3_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.217038    |
|     42 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -6.1967      |
|     42 | M3_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.252249    |
|     42 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -0.778719    |
|     42 | M3_circ         | test_id_ellip | log_omega | poly2    | log_r0         |     -1.49503     |
|     42 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.0441399   |
|     42 | M3_circ         | test_id_ellip | log_omega | poly2    | log_v0         |     -1.4888      |
|     42 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |     -1.7612      |
|     42 | M3_circ         | test_id_ellip | log_omega | poly2    | log_a          |     -3.05822     |
|     42 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |     -7.3175      |
|     42 | M3_circ         | test_id_ellip | log_omega | poly2    | e              |     -5.29342     |
|     42 | M3_circ         | test_ood_ecc  | log_omega | poly2    | e              |   -164.827       |
|     42 | M3_circ         | test_id_ellip | log_omega | poly2    | log_h          |     -2.0445      |
|     42 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |    -17.0953      |
|     42 | M3_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |     -3.05822     |
|     42 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -7.3175      |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_r0         |      0.435392    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_r0         |      0.279205    |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_v0         |      0.197695    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.384605    |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_a          |      0.650484    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_a          |      0.748481    |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | e              |     -0.00473177  |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | e              |   -113.223       |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_h          |      0.534044    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_h          |     -2.86915     |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_abs_energy |      0.650484    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.748481    |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_r0         |      0.442746    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_r0         |      0.213373    |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_v0         |      0.201444    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.335946    |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_a          |      0.735183    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_a          |      0.338587    |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | e              |     -0.0147898   |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | e              |   -110.029       |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_h          |      0.610177    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_h          |     -2.74868     |
|     42 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_abs_energy |      0.735183    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.338587    |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_r0         |     -3.26783     |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_r0         |     -1.15097     |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_v0         |     -1.41929     |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_v0         |     -0.494472    |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_a          |     -4.86114     |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_a          |     -3.10817     |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | e              |     -0.732722    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | e              |   -121.796       |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_h          |     -4.86641     |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_h          |     -9.50372     |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_abs_energy |     -4.86114     |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_abs_energy |     -3.10817     |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_r0         |      0.962533    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_r0         |      0.994371    |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_v0         |      0.948019    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_v0         |      0.96276     |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_a          |      0.981469    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_a          |      0.638265    |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | e              |     -0.988362    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | e              |    -48.5656      |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_h          |      0.868933    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_h          |      0.853962    |
|     42 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_abs_energy |      0.981469    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.638265    |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_r0         |      0.93231     |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.752523    |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_v0         |      0.894929    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.497325    |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_a          |      0.876       |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_a          |     -0.222222    |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | e              |      0.498347    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | e              |    -24.5402      |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_h          |      0.838838    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_h          |     -0.156818    |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.876       |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -0.222222    |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_r0         |      0.964757    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.853382    |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_v0         |      0.818907    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.865042    |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_a          |      0.891053    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_a          |     -0.0594891   |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | e              |      0.16676     |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | e              |   -294.368       |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_h          |      0.851299    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_h          |      0.140478    |
|     42 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.891053    |
|     42 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -0.0594891   |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_r0         |      0.376836    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.749184    |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_v0         |      0.0422823   |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_v0         |     -1.0577      |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_a          |      0.826267    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_a          |     -0.0537477   |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | e              |      0.0146931   |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | e              |   -105.055       |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_h          |      0.817817    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_h          |     -1.24628     |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_abs_energy |      0.826267    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -0.0537477   |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_r0         |      0.363244    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.90491     |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_v0         |      0.110823    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.590537    |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_a          |      0.742625    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_a          |     -2.18669     |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | e              |      0.0541551   |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | e              |    -89.2342      |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_h          |      0.763661    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_h          |     -2.10711     |
|     42 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_abs_energy |      0.742625    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -2.18669     |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_r0         |     -0.626408    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_r0         |     -0.351953    |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_v0         |     -0.958598    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_v0         |     -0.255974    |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_a          |     -1.03615     |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_a          |     -2.46829     |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | e              |     -5.13522     |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | e              |   -232.608       |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_h          |     -1.50369     |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_h          |    -11.4962      |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_abs_energy |     -1.03615     |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.46829     |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_r0         |      0.998665    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_r0         |      0.799948    |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_v0         |      0.991372    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_v0         |      0.900184    |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_a          |      0.852763    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_a          |      0.198478    |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | e              |     -4.57281     |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | e              |   -142.949       |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_h          |      0.185594    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_h          |      0.359941    |
|     42 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_abs_energy |      0.852763    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.198478    |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_r0         |      0.510686    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.436575    |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_v0         |      0.0703241   |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.190954    |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_a          |      0.990024    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_a          |      0.963162    |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | e              |      0.700419    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | e              |    -29.882       |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_h          |      0.942066    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_h          |      0.1872      |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.990024    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.963162    |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_r0         |      0.439502    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_r0         |     -2.44826     |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_v0         |     -0.227931    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_v0         |     -3.6156      |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_a          |      0.988692    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_a          |      0.839342    |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | e              |     -2.08171     |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | e              |   -179.478       |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_h          |      0.945427    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.454467    |
|     42 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.988692    |
|     42 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.839342    |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_r0         |      0.37759     |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.557422    |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_v0         |      0.0346053   |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.939129    |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_a          |      0.850141    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_a          |      0.274924    |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | e              |      0.0108574   |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | e              |   -106.429       |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_h          |      0.819482    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_h          |     -1.2833      |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_abs_energy |      0.850141    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.274924    |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_r0         |      0.370646    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.592619    |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_v0         |      0.111638    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.662523    |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_a          |      0.772285    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_a          |     -0.51344     |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | e              |      0.0394643   |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | e              |    -96.8117      |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_h          |      0.763101    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_h          |     -1.36095     |
|     42 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_abs_energy |      0.772285    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -0.51344     |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_r0         |     -1.67453     |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_r0         |     -0.161815    |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_v0         |     -2.66277     |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_v0         |     -0.143445    |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_a          |     -1.58844     |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_a          |     -2.00323     |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | e              |     -4.57814     |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | e              |   -191.304       |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_h          |     -0.520238    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_h          |     -9.4935      |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_abs_energy |     -1.58844     |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.00323     |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_r0         |      0.99643     |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_r0         |      0.996165    |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_v0         |      0.989572    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_v0         |      0.911226    |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_a          |      0.320017    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_a          |      0.735915    |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | e              |     -1.21758     |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | e              |   -132.52        |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_h          |      0.983952    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_h          |      0.798433    |
|     42 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_abs_energy |      0.320017    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.735915    |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_r0         |      0.516771    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_r0         |     -0.00341853  |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_v0         |      0.0073377   |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_v0         |     -0.569405    |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_a          |      0.996366    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_a          |      0.913359    |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | e              |      0.320252    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | e              |    -17.2306      |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_h          |      0.918423    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_h          |      0.46257     |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.996366    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.913359    |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_r0         |      0.233747    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_r0         |     -3.46613     |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_v0         |     -0.382469    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_v0         |     -5.5867      |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_a          |      0.986566    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_a          |      0.710464    |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | e              |      0.62426     |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | e              |   -217.929       |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_h          |      0.897984    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_h          |      0.115098    |
|     42 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.986566    |
|     42 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.710464    |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_r0         |     -0.834804    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.127603    |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_v0         |     -0.349408    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.0949922   |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_a          |     -1.33482     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_a          |     -1.33482     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | e              |     -0.002616    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | e              |   -116.28        |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_h          |     -1.07162     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_h          |     -7.95375     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_abs_energy |     -1.33482     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -1.33482     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_r0         |     -0.834804    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.127603    |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_v0         |     -0.349408    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0949922   |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_a          |     -1.33482     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_a          |     -1.33482     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | e              |     -0.002616    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | e              |   -116.28        |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_h          |     -1.07162     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_h          |     -7.95375     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_abs_energy |     -1.33482     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.33482     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_r0         |     -0.824139    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_r0         |     -1.34578     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_v0         |     -0.394153    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_v0         |     -0.659275    |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_a          |     -1.20473     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_a          |     -3.13005     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | e              |     -0.404177    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | e              |   -119.385       |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_h          |     -1.14434     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_h          |     -9.44586     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_abs_energy |     -1.20473     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_abs_energy |     -3.13005     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_r0         |      0.780389    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_r0         |      0.68218     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_v0         |      0.800666    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_v0         |      0.742595    |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_a          |      0.396266    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_a          |     -0.832212    |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | e              |      0.269855    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | e              |    -30.193       |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_h          |      0.50752     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_h          |     -4.31768     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_abs_energy |      0.396266    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.832212    |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_r0         |     -0.88234     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_r0         |     -0.298324    |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_v0         |     -0.579349    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.00272146  |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_a          |     -1.22739     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_a          |     -2.16773     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | e              |     -0.432763    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | e              |   -121.228       |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_h          |     -0.884882    |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_h          |     -9.68777     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_abs_energy |     -1.22739     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -2.16773     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_r0         |    -25.341       |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_r0         |     -5.80862     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_v0         |    -20.29        |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_v0         |     -3.60429     |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_a          |    -24.7363      |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_a          |    -11.2208      |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | e              |     -5.39092     |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | e              |   -144.812       |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_h          |    -19.5151      |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_h          |    -15.8207      |
|     42 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_abs_energy |    -24.7363      |
|     42 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_abs_energy |    -11.2208      |
|     43 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_r0         |      0.110157    |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.0310341   |
|     43 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_v0         |      0.128143    |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.0890764   |
|     43 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_a          |      0.13527     |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_a          |      0.421379    |
|     43 | M0_Koopman      | test_id_ellip | norm      | ridge    | e              |      0.0409433   |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | e              |   -124.156       |
|     43 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_h          |     -0.0336038   |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_h          |     -3.89098     |
|     43 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_abs_energy |      0.13527     |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.421379    |
|     43 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_r0         |      0.0792688   |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.14168     |
|     43 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_v0         |      0.131382    |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0536406   |
|     43 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_a          |      0.0266527   |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_a          |      0.309686    |
|     43 | M0_Koopman      | test_id_ellip | norm      | poly2    | e              |     -0.0344934   |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | e              |   -121.747       |
|     43 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_h          |     -0.223735    |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_h          |     -3.78679     |
|     43 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_abs_energy |      0.0266527   |
|     43 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.309686    |
|     43 | M0_Koopman      | test_id_ellip | state     | ridge    | log_r0         |     -1.94119     |
|     43 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_r0         |     -0.302896    |
|     43 | M0_Koopman      | test_id_ellip | state     | ridge    | log_v0         |     -1.29187     |
|     43 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_v0         |      0.141709    |
|     43 | M0_Koopman      | test_id_ellip | state     | ridge    | log_a          |     -1.97674     |
|     43 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_a          |     -1.26819     |
|     43 | M0_Koopman      | test_id_ellip | state     | ridge    | e              |     -0.154134    |
|     43 | M0_Koopman      | test_ood_ecc  | state     | ridge    | e              |   -107.103       |
|     43 | M0_Koopman      | test_id_ellip | state     | ridge    | log_h          |     -1.83562     |
|     43 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_h          |     -6.9865      |
|     43 | M0_Koopman      | test_id_ellip | state     | ridge    | log_abs_energy |     -1.97674     |
|     43 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.26819     |
|     43 | M0_Koopman      | test_id_ellip | state     | poly2    | log_r0         |      0.995026    |
|     43 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_r0         |      0.987266    |
|     43 | M0_Koopman      | test_id_ellip | state     | poly2    | log_v0         |      0.955427    |
|     43 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_v0         |      0.956012    |
|     43 | M0_Koopman      | test_id_ellip | state     | poly2    | log_a          |      0.93567     |
|     43 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_a          |      0.825183    |
|     43 | M0_Koopman      | test_id_ellip | state     | poly2    | e              |     -0.634487    |
|     43 | M0_Koopman      | test_ood_ecc  | state     | poly2    | e              |    -87.427       |
|     43 | M0_Koopman      | test_id_ellip | state     | poly2    | log_h          |      0.98644     |
|     43 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_h          |      0.381231    |
|     43 | M0_Koopman      | test_id_ellip | state     | poly2    | log_abs_energy |      0.93567     |
|     43 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.825183    |
|     43 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_r0         |      0.932272    |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.875874    |
|     43 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_v0         |      0.911958    |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.841439    |
|     43 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_a          |      0.683023    |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_a          |      0.270357    |
|     43 | M0_Koopman      | test_id_ellip | log_omega | ridge    | e              |     -0.480981    |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | e              |    -66.9596      |
|     43 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_h          |      0.358887    |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_h          |     -1.58011     |
|     43 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.683023    |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.270357    |
|     43 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_r0         |      0.976887    |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.926109    |
|     43 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_v0         |      0.616774    |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.888488    |
|     43 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_a          |      0.692754    |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_a          |     -0.074455    |
|     43 | M0_Koopman      | test_id_ellip | log_omega | poly2    | e              |     -5.77837     |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | e              |   -387.047       |
|     43 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_h          |      0.405493    |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_h          |      0.395558    |
|     43 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.692754    |
|     43 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -0.074455    |
|     43 | M1_circ         | test_id_ellip | norm      | ridge    | log_r0         |      0.121815    |
|     43 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.0295299   |
|     43 | M1_circ         | test_id_ellip | norm      | ridge    | log_v0         |      0.140699    |
|     43 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.089474    |
|     43 | M1_circ         | test_id_ellip | norm      | ridge    | log_a          |      0.140815    |
|     43 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_a          |      0.413512    |
|     43 | M1_circ         | test_id_ellip | norm      | ridge    | e              |      0.0410259   |
|     43 | M1_circ         | test_ood_ecc  | norm      | ridge    | e              |   -124.486       |
|     43 | M1_circ         | test_id_ellip | norm      | ridge    | log_h          |     -0.030288    |
|     43 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -3.83415     |
|     43 | M1_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |      0.140815    |
|     43 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.413512    |
|     43 | M1_circ         | test_id_ellip | norm      | poly2    | log_r0         |      0.0907127   |
|     43 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.148719    |
|     43 | M1_circ         | test_id_ellip | norm      | poly2    | log_v0         |      0.142931    |
|     43 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0570736   |
|     43 | M1_circ         | test_id_ellip | norm      | poly2    | log_a          |      0.0306278   |
|     43 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_a          |      0.280477    |
|     43 | M1_circ         | test_id_ellip | norm      | poly2    | e              |     -0.0358007   |
|     43 | M1_circ         | test_ood_ecc  | norm      | poly2    | e              |   -121.614       |
|     43 | M1_circ         | test_id_ellip | norm      | poly2    | log_h          |     -0.22451     |
|     43 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -3.70539     |
|     43 | M1_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |      0.0306278   |
|     43 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.280477    |
|     43 | M1_circ         | test_id_ellip | state     | ridge    | log_r0         |     -2.80603     |
|     43 | M1_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -0.556915    |
|     43 | M1_circ         | test_id_ellip | state     | ridge    | log_v0         |     -1.81749     |
|     43 | M1_circ         | test_ood_ecc  | state     | ridge    | log_v0         |      0.0205763   |
|     43 | M1_circ         | test_id_ellip | state     | ridge    | log_a          |     -2.95296     |
|     43 | M1_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -1.58143     |
|     43 | M1_circ         | test_id_ellip | state     | ridge    | e              |     -0.228208    |
|     43 | M1_circ         | test_ood_ecc  | state     | ridge    | e              |   -102.819       |
|     43 | M1_circ         | test_id_ellip | state     | ridge    | log_h          |     -2.82048     |
|     43 | M1_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -7.05873     |
|     43 | M1_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -2.95296     |
|     43 | M1_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.58143     |
|     43 | M1_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.938772    |
|     43 | M1_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.841703    |
|     43 | M1_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.983231    |
|     43 | M1_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.929669    |
|     43 | M1_circ         | test_id_ellip | state     | poly2    | log_a          |      0.703937    |
|     43 | M1_circ         | test_ood_ecc  | state     | poly2    | log_a          |      0.706899    |
|     43 | M1_circ         | test_id_ellip | state     | poly2    | e              |     -3.5943      |
|     43 | M1_circ         | test_ood_ecc  | state     | poly2    | e              |    -73.3474      |
|     43 | M1_circ         | test_id_ellip | state     | poly2    | log_h          |      0.955125    |
|     43 | M1_circ         | test_ood_ecc  | state     | poly2    | log_h          |      0.723124    |
|     43 | M1_circ         | test_id_ellip | state     | poly2    | log_abs_energy |      0.703937    |
|     43 | M1_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.706899    |
|     43 | M1_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.921403    |
|     43 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.813396    |
|     43 | M1_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.961811    |
|     43 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.954113    |
|     43 | M1_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.72506     |
|     43 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |     -0.175347    |
|     43 | M1_circ         | test_id_ellip | log_omega | ridge    | e              |      0.232109    |
|     43 | M1_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -68.1304      |
|     43 | M1_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.56777     |
|     43 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -1.67781     |
|     43 | M1_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.72506     |
|     43 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -0.175347    |
|     43 | M1_circ         | test_id_ellip | log_omega | poly2    | log_r0         |      0.917455    |
|     43 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.805228    |
|     43 | M1_circ         | test_id_ellip | log_omega | poly2    | log_v0         |      0.981853    |
|     43 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.97827     |
|     43 | M1_circ         | test_id_ellip | log_omega | poly2    | log_a          |      0.296223    |
|     43 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |     -0.581714    |
|     43 | M1_circ         | test_id_ellip | log_omega | poly2    | e              |      0.448864    |
|     43 | M1_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -32.902       |
|     43 | M1_circ         | test_id_ellip | log_omega | poly2    | log_h          |      0.396618    |
|     43 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -1.5064      |
|     43 | M1_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.296223    |
|     43 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -0.581714    |
|     43 | M2_circ         | test_id_ellip | norm      | ridge    | log_r0         |      0.102826    |
|     43 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.0348925   |
|     43 | M2_circ         | test_id_ellip | norm      | ridge    | log_v0         |      0.136666    |
|     43 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.0737406   |
|     43 | M2_circ         | test_id_ellip | norm      | ridge    | log_a          |      0.104626    |
|     43 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_a          |      0.362358    |
|     43 | M2_circ         | test_id_ellip | norm      | ridge    | e              |      0.046975    |
|     43 | M2_circ         | test_ood_ecc  | norm      | ridge    | e              |   -124.259       |
|     43 | M2_circ         | test_id_ellip | norm      | ridge    | log_h          |     -0.0699016   |
|     43 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -4.05302     |
|     43 | M2_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |      0.104626    |
|     43 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.362358    |
|     43 | M2_circ         | test_id_ellip | norm      | poly2    | log_r0         |      0.0674472   |
|     43 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.149967    |
|     43 | M2_circ         | test_id_ellip | norm      | poly2    | log_v0         |      0.139468    |
|     43 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0468025   |
|     43 | M2_circ         | test_id_ellip | norm      | poly2    | log_a          |     -0.0207694   |
|     43 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_a          |      0.231833    |
|     43 | M2_circ         | test_id_ellip | norm      | poly2    | e              |     -0.0269839   |
|     43 | M2_circ         | test_ood_ecc  | norm      | poly2    | e              |   -122.629       |
|     43 | M2_circ         | test_id_ellip | norm      | poly2    | log_h          |     -0.281914    |
|     43 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -4.0281      |
|     43 | M2_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |     -0.0207694   |
|     43 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.231833    |
|     43 | M2_circ         | test_id_ellip | state     | ridge    | log_r0         |     -3.71972     |
|     43 | M2_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -1.42197     |
|     43 | M2_circ         | test_id_ellip | state     | ridge    | log_v0         |     -2.31613     |
|     43 | M2_circ         | test_ood_ecc  | state     | ridge    | log_v0         |     -0.214885    |
|     43 | M2_circ         | test_id_ellip | state     | ridge    | log_a          |     -4.34516     |
|     43 | M2_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -3.45955     |
|     43 | M2_circ         | test_id_ellip | state     | ridge    | e              |     -0.0251152   |
|     43 | M2_circ         | test_ood_ecc  | state     | ridge    | e              |   -106.492       |
|     43 | M2_circ         | test_id_ellip | state     | ridge    | log_h          |     -3.6615      |
|     43 | M2_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -8.44536     |
|     43 | M2_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -4.34516     |
|     43 | M2_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -3.45955     |
|     43 | M2_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.985648    |
|     43 | M2_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.96874     |
|     43 | M2_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.983957    |
|     43 | M2_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.965766    |
|     43 | M2_circ         | test_id_ellip | state     | poly2    | log_a          |      0.866012    |
|     43 | M2_circ         | test_ood_ecc  | state     | poly2    | log_a          |      0.182716    |
|     43 | M2_circ         | test_id_ellip | state     | poly2    | e              |     -1.4596      |
|     43 | M2_circ         | test_ood_ecc  | state     | poly2    | e              |   -174.898       |
|     43 | M2_circ         | test_id_ellip | state     | poly2    | log_h          |      0.988456    |
|     43 | M2_circ         | test_ood_ecc  | state     | poly2    | log_h          |      0.778569    |
|     43 | M2_circ         | test_id_ellip | state     | poly2    | log_abs_energy |      0.866012    |
|     43 | M2_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.182716    |
|     43 | M2_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.925224    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.873383    |
|     43 | M2_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.951274    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.891442    |
|     43 | M2_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.795744    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |     -0.0187557   |
|     43 | M2_circ         | test_id_ellip | log_omega | ridge    | e              |      0.364452    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -43.2987      |
|     43 | M2_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.778685    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -0.748547    |
|     43 | M2_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.795744    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -0.0187557   |
|     43 | M2_circ         | test_id_ellip | log_omega | poly2    | log_r0         |      0.915303    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.866224    |
|     43 | M2_circ         | test_id_ellip | log_omega | poly2    | log_v0         |      0.953508    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.901249    |
|     43 | M2_circ         | test_id_ellip | log_omega | poly2    | log_a          |      0.645635    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |      0.147588    |
|     43 | M2_circ         | test_id_ellip | log_omega | poly2    | e              |      0.132969    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -56.1875      |
|     43 | M2_circ         | test_id_ellip | log_omega | poly2    | log_h          |      0.753729    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.711076    |
|     43 | M2_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.645635    |
|     43 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.147588    |
|     43 | M3_circ         | test_id_ellip | norm      | ridge    | log_r0         |     -1.16024     |
|     43 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.57206     |
|     43 | M3_circ         | test_id_ellip | norm      | ridge    | log_v0         |     -0.809591    |
|     43 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.0156489   |
|     43 | M3_circ         | test_id_ellip | norm      | ridge    | log_a          |     -1.33482     |
|     43 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_a          |     -1.33482     |
|     43 | M3_circ         | test_id_ellip | norm      | ridge    | e              |     -7.07791e-05 |
|     43 | M3_circ         | test_ood_ecc  | norm      | ridge    | e              |   -110.909       |
|     43 | M3_circ         | test_id_ellip | norm      | ridge    | log_h          |     -1.04922     |
|     43 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -7.89201     |
|     43 | M3_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |     -1.33482     |
|     43 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -1.33482     |
|     43 | M3_circ         | test_id_ellip | norm      | poly2    | log_r0         |     -1.16024     |
|     43 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.57206     |
|     43 | M3_circ         | test_id_ellip | norm      | poly2    | log_v0         |     -0.809591    |
|     43 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0156489   |
|     43 | M3_circ         | test_id_ellip | norm      | poly2    | log_a          |     -1.33482     |
|     43 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_a          |     -1.33482     |
|     43 | M3_circ         | test_id_ellip | norm      | poly2    | e              |     -7.07791e-05 |
|     43 | M3_circ         | test_ood_ecc  | norm      | poly2    | e              |   -110.909       |
|     43 | M3_circ         | test_id_ellip | norm      | poly2    | log_h          |     -1.04922     |
|     43 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -7.89201     |
|     43 | M3_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |     -1.33482     |
|     43 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.33482     |
|     43 | M3_circ         | test_id_ellip | state     | ridge    | log_r0         |     -3.2478      |
|     43 | M3_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -0.623244    |
|     43 | M3_circ         | test_id_ellip | state     | ridge    | log_v0         |     -2.32786     |
|     43 | M3_circ         | test_ood_ecc  | state     | ridge    | log_v0         |      0.0695851   |
|     43 | M3_circ         | test_id_ellip | state     | ridge    | log_a          |     -3.9823      |
|     43 | M3_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -2.26389     |
|     43 | M3_circ         | test_id_ellip | state     | ridge    | e              |     -0.373381    |
|     43 | M3_circ         | test_ood_ecc  | state     | ridge    | e              |    -86.5821      |
|     43 | M3_circ         | test_id_ellip | state     | ridge    | log_h          |     -3.35989     |
|     43 | M3_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -7.72901     |
|     43 | M3_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -3.9823      |
|     43 | M3_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.26389     |
|     43 | M3_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.779127    |
|     43 | M3_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.707434    |
|     43 | M3_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.597147    |
|     43 | M3_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.73882     |
|     43 | M3_circ         | test_id_ellip | state     | poly2    | log_a          |      0.322006    |
|     43 | M3_circ         | test_ood_ecc  | state     | poly2    | log_a          |     -0.39622     |
|     43 | M3_circ         | test_id_ellip | state     | poly2    | e              |     -0.0406195   |
|     43 | M3_circ         | test_ood_ecc  | state     | poly2    | e              |    -44.6585      |
|     43 | M3_circ         | test_id_ellip | state     | poly2    | log_h          |     -0.0242749   |
|     43 | M3_circ         | test_ood_ecc  | state     | poly2    | log_h          |     -2.4613      |
|     43 | M3_circ         | test_id_ellip | state     | poly2    | log_abs_energy |      0.322006    |
|     43 | M3_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.39622     |
|     43 | M3_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.684823    |
|     43 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.589354    |
|     43 | M3_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.700521    |
|     43 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.499642    |
|     43 | M3_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.264571    |
|     43 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |     -0.66458     |
|     43 | M3_circ         | test_id_ellip | log_omega | ridge    | e              |      0.112164    |
|     43 | M3_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -90.3853      |
|     43 | M3_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.170389    |
|     43 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -4.61082     |
|     43 | M3_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.264571    |
|     43 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -0.66458     |
|     43 | M3_circ         | test_id_ellip | log_omega | poly2    | log_r0         |     -0.993133    |
|     43 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.442774    |
|     43 | M3_circ         | test_id_ellip | log_omega | poly2    | log_v0         |      0.0979781   |
|     43 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.386304    |
|     43 | M3_circ         | test_id_ellip | log_omega | poly2    | log_a          |     -8.90442     |
|     43 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |     -2.58336     |
|     43 | M3_circ         | test_id_ellip | log_omega | poly2    | e              |     -1.40221     |
|     43 | M3_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -76.9499      |
|     43 | M3_circ         | test_id_ellip | log_omega | poly2    | log_h          |     -8.4569      |
|     43 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -5.63422     |
|     43 | M3_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |     -8.90442     |
|     43 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -2.58336     |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_r0         |      0.519496    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_r0         |      0.0495631   |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_v0         |      0.367307    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.204894    |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_a          |      0.609127    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_a          |      0.595184    |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | e              |      0.00259687  |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | e              |   -126.624       |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_h          |      0.437726    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_h          |     -2.50191     |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_abs_energy |      0.609127    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.595184    |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_r0         |      0.585566    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.12835     |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_v0         |      0.342798    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.159665    |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_a          |      0.739528    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_a          |      0.205295    |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | e              |     -0.026532    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | e              |   -117.409       |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_h          |      0.545694    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_h          |     -2.03615     |
|     43 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_abs_energy |      0.739528    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.205295    |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_r0         |     -1.5065      |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_r0         |     -0.442185    |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_v0         |     -1.07174     |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_v0         |      0.077857    |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_a          |     -1.47773     |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_a          |     -1.30802     |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | e              |     -0.230662    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | e              |   -107.855       |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_h          |     -1.42795     |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_h          |     -7.3156      |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_abs_energy |     -1.47773     |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.30802     |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_r0         |      0.986281    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_r0         |      0.974004    |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_v0         |      0.979064    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_v0         |      0.951934    |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_a          |      0.986023    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_a          |      0.90389     |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | e              |      0.61463     |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | e              |    -66.8958      |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_h          |      0.958672    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_h          |      0.655568    |
|     43 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_abs_energy |      0.986023    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.90389     |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_r0         |      0.941322    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.725203    |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_v0         |      0.905374    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.818052    |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_a          |      0.956611    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_a          |      0.580423    |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | e              |      0.189144    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | e              |    -31.9305      |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_h          |      0.833297    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_h          |      0.0898973   |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.956611    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.580423    |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_r0         |      0.964363    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.728929    |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_v0         |      0.941566    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.783258    |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_a          |      0.978842    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_a          |      0.654419    |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | e              |      0.568429    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | e              |    -38.3767      |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_h          |      0.951044    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_h          |      0.615088    |
|     43 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.978842    |
|     43 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.654419    |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_r0         |      0.382644    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.424665    |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_v0         |      0.110982    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.537974    |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_a          |      0.750345    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_a          |      0.417763    |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | e              |     -0.0183045   |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | e              |   -120.155       |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_h          |      0.722886    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_h          |     -1.29102     |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_abs_energy |      0.750345    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.417763    |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_r0         |      0.386648    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.482841    |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_v0         |      0.0781293   |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.202137    |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_a          |      0.802602    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_a          |     -0.544328    |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | e              |      0.035397    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | e              |    -99.6595      |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_h          |      0.819952    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_h          |     -1.31868     |
|     43 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_abs_energy |      0.802602    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -0.544328    |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_r0         |     -2.26679     |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_r0         |     -2.7947      |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_v0         |     -1.52226     |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_v0         |     -0.811427    |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_a          |     -2.60711     |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_a          |     -5.5885      |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | e              |     -0.405631    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | e              |   -112.901       |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_h          |     -2.60366     |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_h          |    -11.1508      |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_abs_energy |     -2.60711     |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_abs_energy |     -5.5885      |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_r0         |      0.999293    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_r0         |      0.990623    |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_v0         |      0.998124    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_v0         |      0.981385    |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_a          |      0.9811      |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_a          |      0.929417    |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | e              |     -0.306327    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | e              |   -127.029       |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_h          |      0.94361     |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_h          |      0.68788     |
|     43 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_abs_energy |      0.9811      |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.929417    |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_r0         |      0.754941    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.313756    |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_v0         |      0.615712    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.235509    |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_a          |      0.992295    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_a          |      0.890671    |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | e              |      0.708544    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | e              |    -32.2188      |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_h          |      0.956596    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_h          |      0.0408847   |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.992295    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.890671    |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_r0         |      0.593133    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_r0         |     -9.85832     |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_v0         |     -0.0784063   |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_v0         |    -12.1232      |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_a          |      0.968688    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_a          |      0.491701    |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | e              |     -0.0847636   |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | e              |   -342.715       |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_h          |      0.892737    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.46167     |
|     43 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.968688    |
|     43 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.491701    |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_r0         |      0.394879    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.445582    |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_v0         |      0.111244    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.531363    |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_a          |      0.760176    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_a          |      0.278467    |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | e              |     -0.01777     |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | e              |   -118.448       |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_h          |      0.757272    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_h          |     -1.47302     |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_abs_energy |      0.760176    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.278467    |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_r0         |      0.392587    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.379224    |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_v0         |      0.0789148   |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.162429    |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_a          |      0.757948    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_a          |     -0.508422    |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | e              |      0.0464356   |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | e              |   -101.083       |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_h          |      0.784962    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_h          |     -1.68434     |
|     43 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_abs_energy |      0.757948    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -0.508422    |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_r0         |     -6.45875     |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_r0         |     -7.28843     |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_v0         |     -4.79826     |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_v0         |     -2.69656     |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_a          |     -6.90207     |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_a          |    -12.9418      |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | e              |     -0.252851    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | e              |   -137.172       |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_h          |     -5.59623     |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_h          |    -14.3612      |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_abs_energy |     -6.90207     |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_abs_energy |    -12.9418      |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_r0         |      0.999795    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_r0         |      0.998493    |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_v0         |      0.994497    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_v0         |      0.987653    |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_a          |      0.947514    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_a          |      0.00878126  |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | e              |      0.0389912   |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | e              |   -169.363       |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_h          |      0.982802    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_h          |      0.871368    |
|     43 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_abs_energy |      0.947514    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.00878126  |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_r0         |      0.572233    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.379458    |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_v0         |      0.213704    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.38971     |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_a          |      0.988307    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_a          |      0.886365    |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | e              |      0.595816    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | e              |    -35.35        |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_h          |      0.962724    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_h          |     -0.180063    |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.988307    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.886365    |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_r0         |      0.808129    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_r0         |     -0.1472      |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_v0         |      0.614354    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.113185    |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_a          |      0.994624    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_a          |      0.870834    |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | e              |      0.437039    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | e              |   -325.798       |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_h          |      0.862106    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_h          |     -2.13633     |
|     43 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.994624    |
|     43 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.870834    |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_r0         |     -1.16024     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.57206     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_v0         |     -0.809591    |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.0156489   |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_a          |     -1.33482     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_a          |     -1.33482     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | e              |     -7.07791e-05 |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | e              |   -110.909       |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_h          |     -1.04922     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_h          |     -7.89201     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_abs_energy |     -1.33482     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -1.33482     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_r0         |     -1.16024     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.57206     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_v0         |     -0.809591    |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0156489   |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_a          |     -1.33482     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_a          |     -1.33482     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | e              |     -7.07791e-05 |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | e              |   -110.909       |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_h          |     -1.04922     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_h          |     -7.89201     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_abs_energy |     -1.33482     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.33482     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_r0         |     -1.11805     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_r0         |     -1.13182     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_v0         |     -0.931021    |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_v0         |     -0.307975    |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_a          |     -1.23616     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_a          |     -1.89082     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | e              |     -0.403411    |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | e              |    -99.9568      |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_h          |     -1.14187     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_h          |     -7.91696     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_abs_energy |     -1.23616     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.89082     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_r0         |      0.817445    |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_r0         |      0.525946    |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_v0         |      0.80981     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_v0         |      0.715539    |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_a          |      0.674367    |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_a          |     -0.533066    |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | e              |      0.170206    |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | e              |    -30.4661      |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_h          |      0.613178    |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_h          |     -3.58165     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_abs_energy |      0.674367    |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.533066    |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_r0         |     -1.20523     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_r0         |     -0.512998    |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_v0         |     -0.650565    |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_v0         |     -0.119611    |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_a          |     -1.7833      |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_a          |     -1.04089     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | e              |     -0.226809    |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | e              |   -107.377       |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_h          |     -1.67335     |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_h          |     -7.17842     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_abs_energy |     -1.7833      |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -1.04089     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_r0         |    -32.7516      |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_r0         |     -4.87526     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_v0         |    -20.6195      |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_v0         |     -3.11337     |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_a          |    -48.1284      |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_a          |    -14.9189      |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | e              |    -13.6424      |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | e              |   -459.589       |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_h          |    -43.5719      |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_h          |    -16.1808      |
|     43 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_abs_energy |    -48.1284      |
|     43 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_abs_energy |    -14.9189      |
|     44 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_r0         |      0.0894024   |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_r0         |      0.211427    |
|     44 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_v0         |     -0.0419864   |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.225209    |
|     44 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_a          |      0.287373    |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_a          |      0.584896    |
|     44 | M0_Koopman      | test_id_ellip | norm      | ridge    | e              |      0.00830875  |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | e              |   -118.733       |
|     44 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_h          |      0.181858    |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_h          |     -3.63341     |
|     44 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_abs_energy |      0.287373    |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.584896    |
|     44 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_r0         |      0.0827906   |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_r0         |      0.20199     |
|     44 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_v0         |      0.0351838   |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.16036     |
|     44 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_a          |      0.236668    |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_a          |      0.520608    |
|     44 | M0_Koopman      | test_id_ellip | norm      | poly2    | e              |      0.00456103  |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | e              |   -118.13        |
|     44 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_h          |      0.12402     |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_h          |     -3.51237     |
|     44 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_abs_energy |      0.236668    |
|     44 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.520608    |
|     44 | M0_Koopman      | test_id_ellip | state     | ridge    | log_r0         |     -1.28835     |
|     44 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_r0         |     -0.52229     |
|     44 | M0_Koopman      | test_id_ellip | state     | ridge    | log_v0         |     -0.835921    |
|     44 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_v0         |     -0.0262856   |
|     44 | M0_Koopman      | test_id_ellip | state     | ridge    | log_a          |     -1.48431     |
|     44 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_a          |     -1.56288     |
|     44 | M0_Koopman      | test_id_ellip | state     | ridge    | e              |      0.0116367   |
|     44 | M0_Koopman      | test_ood_ecc  | state     | ridge    | e              |   -107.129       |
|     44 | M0_Koopman      | test_id_ellip | state     | ridge    | log_h          |     -1.15553     |
|     44 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_h          |     -7.94537     |
|     44 | M0_Koopman      | test_id_ellip | state     | ridge    | log_abs_energy |     -1.48431     |
|     44 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.56288     |
|     44 | M0_Koopman      | test_id_ellip | state     | poly2    | log_r0         |      0.998057    |
|     44 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_r0         |      0.996686    |
|     44 | M0_Koopman      | test_id_ellip | state     | poly2    | log_v0         |      0.993068    |
|     44 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_v0         |      0.956587    |
|     44 | M0_Koopman      | test_id_ellip | state     | poly2    | log_a          |      0.945827    |
|     44 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_a          |      0.934779    |
|     44 | M0_Koopman      | test_id_ellip | state     | poly2    | e              |      0.569346    |
|     44 | M0_Koopman      | test_ood_ecc  | state     | poly2    | e              |    -27.2398      |
|     44 | M0_Koopman      | test_id_ellip | state     | poly2    | log_h          |      0.984553    |
|     44 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_h          |      0.751055    |
|     44 | M0_Koopman      | test_id_ellip | state     | poly2    | log_abs_energy |      0.945827    |
|     44 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.934779    |
|     44 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_r0         |      0.876778    |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.911829    |
|     44 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_v0         |      0.803914    |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.608904    |
|     44 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_a          |      0.810164    |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_a          |      0.448261    |
|     44 | M0_Koopman      | test_id_ellip | log_omega | ridge    | e              |      0.128904    |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | e              |    -45.9666      |
|     44 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_h          |      0.841613    |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_h          |     -1.19821     |
|     44 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.810164    |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.448261    |
|     44 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_r0         |      0.919079    |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.861833    |
|     44 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_v0         |      0.293547    |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_v0         |     -0.400733    |
|     44 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_a          |      0.637807    |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_a          |     -8.543       |
|     44 | M0_Koopman      | test_id_ellip | log_omega | poly2    | e              |    -52.0244      |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | e              |   -904.486       |
|     44 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_h          |     -1.51372     |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_h          |     -4.23525     |
|     44 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.637807    |
|     44 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -8.543       |
|     44 | M1_circ         | test_id_ellip | norm      | ridge    | log_r0         |      0.121116    |
|     44 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |      0.243133    |
|     44 | M1_circ         | test_id_ellip | norm      | ridge    | log_v0         |     -0.0186947   |
|     44 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.201928    |
|     44 | M1_circ         | test_id_ellip | norm      | ridge    | log_a          |      0.316189    |
|     44 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_a          |      0.584937    |
|     44 | M1_circ         | test_id_ellip | norm      | ridge    | e              |      0.00835798  |
|     44 | M1_circ         | test_ood_ecc  | norm      | ridge    | e              |   -118.628       |
|     44 | M1_circ         | test_id_ellip | norm      | ridge    | log_h          |      0.204168    |
|     44 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -3.70563     |
|     44 | M1_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |      0.316189    |
|     44 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.584937    |
|     44 | M1_circ         | test_id_ellip | norm      | poly2    | log_r0         |      0.115667    |
|     44 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |      0.235831    |
|     44 | M1_circ         | test_id_ellip | norm      | poly2    | log_v0         |      0.056725    |
|     44 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.145374    |
|     44 | M1_circ         | test_id_ellip | norm      | poly2    | log_a          |      0.269889    |
|     44 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_a          |      0.527103    |
|     44 | M1_circ         | test_id_ellip | norm      | poly2    | e              |      0.00505032  |
|     44 | M1_circ         | test_ood_ecc  | norm      | poly2    | e              |   -118.17        |
|     44 | M1_circ         | test_id_ellip | norm      | poly2    | log_h          |      0.151899    |
|     44 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -3.62171     |
|     44 | M1_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |      0.269889    |
|     44 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.527103    |
|     44 | M1_circ         | test_id_ellip | state     | ridge    | log_r0         |     -1.17132     |
|     44 | M1_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -0.320674    |
|     44 | M1_circ         | test_id_ellip | state     | ridge    | log_v0         |     -0.735308    |
|     44 | M1_circ         | test_ood_ecc  | state     | ridge    | log_v0         |      0.0571455   |
|     44 | M1_circ         | test_id_ellip | state     | ridge    | log_a          |     -1.4079      |
|     44 | M1_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -1.44        |
|     44 | M1_circ         | test_id_ellip | state     | ridge    | e              |      0.0200393   |
|     44 | M1_circ         | test_ood_ecc  | state     | ridge    | e              |   -107.321       |
|     44 | M1_circ         | test_id_ellip | state     | ridge    | log_h          |     -1.097       |
|     44 | M1_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -7.86481     |
|     44 | M1_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -1.4079      |
|     44 | M1_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.44        |
|     44 | M1_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.995222    |
|     44 | M1_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.977821    |
|     44 | M1_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.985435    |
|     44 | M1_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.944036    |
|     44 | M1_circ         | test_id_ellip | state     | poly2    | log_a          |      0.947321    |
|     44 | M1_circ         | test_ood_ecc  | state     | poly2    | log_a          |      0.777486    |
|     44 | M1_circ         | test_id_ellip | state     | poly2    | e              |      0.142891    |
|     44 | M1_circ         | test_ood_ecc  | state     | poly2    | e              |    -93.2739      |
|     44 | M1_circ         | test_id_ellip | state     | poly2    | log_h          |      0.988591    |
|     44 | M1_circ         | test_ood_ecc  | state     | poly2    | log_h          |      0.795737    |
|     44 | M1_circ         | test_id_ellip | state     | poly2    | log_abs_energy |      0.947321    |
|     44 | M1_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.777486    |
|     44 | M1_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.986336    |
|     44 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.963074    |
|     44 | M1_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.975659    |
|     44 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.941274    |
|     44 | M1_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.769651    |
|     44 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |      0.578467    |
|     44 | M1_circ         | test_id_ellip | log_omega | ridge    | e              |     -0.0752538   |
|     44 | M1_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -36.4324      |
|     44 | M1_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.708889    |
|     44 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -0.318885    |
|     44 | M1_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.769651    |
|     44 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.578467    |
|     44 | M1_circ         | test_id_ellip | log_omega | poly2    | log_r0         |      0.982686    |
|     44 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.958568    |
|     44 | M1_circ         | test_id_ellip | log_omega | poly2    | log_v0         |      0.975229    |
|     44 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.967224    |
|     44 | M1_circ         | test_id_ellip | log_omega | poly2    | log_a          |      0.862965    |
|     44 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |      0.332964    |
|     44 | M1_circ         | test_id_ellip | log_omega | poly2    | e              |      0.449115    |
|     44 | M1_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -23.6274      |
|     44 | M1_circ         | test_id_ellip | log_omega | poly2    | log_h          |      0.821473    |
|     44 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |      0.246422    |
|     44 | M1_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.862965    |
|     44 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.332964    |
|     44 | M2_circ         | test_id_ellip | norm      | ridge    | log_r0         |      0.0758141   |
|     44 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |      0.256779    |
|     44 | M2_circ         | test_id_ellip | norm      | ridge    | log_v0         |     -0.0412389   |
|     44 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.154545    |
|     44 | M2_circ         | test_id_ellip | norm      | ridge    | log_a          |      0.258732    |
|     44 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_a          |      0.493173    |
|     44 | M2_circ         | test_id_ellip | norm      | ridge    | e              |      0.0132149   |
|     44 | M2_circ         | test_ood_ecc  | norm      | ridge    | e              |   -118.517       |
|     44 | M2_circ         | test_id_ellip | norm      | ridge    | log_h          |      0.142159    |
|     44 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -4.11852     |
|     44 | M2_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |      0.258732    |
|     44 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.493173    |
|     44 | M2_circ         | test_id_ellip | norm      | poly2    | log_r0         |      0.0725885   |
|     44 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |      0.253729    |
|     44 | M2_circ         | test_id_ellip | norm      | poly2    | log_v0         |      0.0390212   |
|     44 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.11596     |
|     44 | M2_circ         | test_id_ellip | norm      | poly2    | log_a          |      0.198278    |
|     44 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_a          |      0.442784    |
|     44 | M2_circ         | test_id_ellip | norm      | poly2    | e              |      0.0126762   |
|     44 | M2_circ         | test_ood_ecc  | norm      | poly2    | e              |   -118.506       |
|     44 | M2_circ         | test_id_ellip | norm      | poly2    | log_h          |      0.0772801   |
|     44 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -4.14318     |
|     44 | M2_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |      0.198278    |
|     44 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.442784    |
|     44 | M2_circ         | test_id_ellip | state     | ridge    | log_r0         |     -1.244       |
|     44 | M2_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -1.03946     |
|     44 | M2_circ         | test_id_ellip | state     | ridge    | log_v0         |     -0.884384    |
|     44 | M2_circ         | test_ood_ecc  | state     | ridge    | log_v0         |     -0.265471    |
|     44 | M2_circ         | test_id_ellip | state     | ridge    | log_a          |     -1.33228     |
|     44 | M2_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -1.8513      |
|     44 | M2_circ         | test_id_ellip | state     | ridge    | e              |     -0.0808265   |
|     44 | M2_circ         | test_ood_ecc  | state     | ridge    | e              |   -104.424       |
|     44 | M2_circ         | test_id_ellip | state     | ridge    | log_h          |     -1.10213     |
|     44 | M2_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -8.26386     |
|     44 | M2_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -1.33228     |
|     44 | M2_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.8513      |
|     44 | M2_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.998825    |
|     44 | M2_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.992691    |
|     44 | M2_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.935445    |
|     44 | M2_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.694916    |
|     44 | M2_circ         | test_id_ellip | state     | poly2    | log_a          |      0.925671    |
|     44 | M2_circ         | test_ood_ecc  | state     | poly2    | log_a          |      0.481403    |
|     44 | M2_circ         | test_id_ellip | state     | poly2    | e              |     -0.612811    |
|     44 | M2_circ         | test_ood_ecc  | state     | poly2    | e              |   -121.409       |
|     44 | M2_circ         | test_id_ellip | state     | poly2    | log_h          |      0.983296    |
|     44 | M2_circ         | test_ood_ecc  | state     | poly2    | log_h          |      0.756842    |
|     44 | M2_circ         | test_id_ellip | state     | poly2    | log_abs_energy |      0.925671    |
|     44 | M2_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.481403    |
|     44 | M2_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.984372    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.972408    |
|     44 | M2_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.950877    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.857005    |
|     44 | M2_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.808659    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |      0.513165    |
|     44 | M2_circ         | test_id_ellip | log_omega | ridge    | e              |      0.300001    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -23.2915      |
|     44 | M2_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.843197    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |      0.0117205   |
|     44 | M2_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.808659    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.513165    |
|     44 | M2_circ         | test_id_ellip | log_omega | poly2    | log_r0         |      0.984037    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.969836    |
|     44 | M2_circ         | test_id_ellip | log_omega | poly2    | log_v0         |      0.977796    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.898084    |
|     44 | M2_circ         | test_id_ellip | log_omega | poly2    | log_a          |      0.906459    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |      0.43503     |
|     44 | M2_circ         | test_id_ellip | log_omega | poly2    | e              |      0.535747    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -10.97        |
|     44 | M2_circ         | test_id_ellip | log_omega | poly2    | log_h          |      0.977183    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |      0.700362    |
|     44 | M2_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.906459    |
|     44 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.43503     |
|     44 | M3_circ         | test_id_ellip | norm      | ridge    | log_r0         |     -1.09993     |
|     44 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.28753     |
|     44 | M3_circ         | test_id_ellip | norm      | ridge    | log_v0         |     -0.693261    |
|     44 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.0287097   |
|     44 | M3_circ         | test_id_ellip | norm      | ridge    | log_a          |     -1.33482     |
|     44 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_a          |     -1.33482     |
|     44 | M3_circ         | test_id_ellip | norm      | ridge    | e              |     -0.000218993 |
|     44 | M3_circ         | test_ood_ecc  | norm      | ridge    | e              |   -110.341       |
|     44 | M3_circ         | test_id_ellip | norm      | ridge    | log_h          |     -1.0263      |
|     44 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -7.82839     |
|     44 | M3_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |     -1.33482     |
|     44 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -1.33482     |
|     44 | M3_circ         | test_id_ellip | norm      | poly2    | log_r0         |     -1.09993     |
|     44 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.28753     |
|     44 | M3_circ         | test_id_ellip | norm      | poly2    | log_v0         |     -0.693261    |
|     44 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0287097   |
|     44 | M3_circ         | test_id_ellip | norm      | poly2    | log_a          |     -1.33482     |
|     44 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_a          |     -1.33482     |
|     44 | M3_circ         | test_id_ellip | norm      | poly2    | e              |     -0.000218993 |
|     44 | M3_circ         | test_ood_ecc  | norm      | poly2    | e              |   -110.341       |
|     44 | M3_circ         | test_id_ellip | norm      | poly2    | log_h          |     -1.0263      |
|     44 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -7.82839     |
|     44 | M3_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |     -1.33482     |
|     44 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.33482     |
|     44 | M3_circ         | test_id_ellip | state     | ridge    | log_r0         |     -4.21465     |
|     44 | M3_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -0.810715    |
|     44 | M3_circ         | test_id_ellip | state     | ridge    | log_v0         |     -3.18103     |
|     44 | M3_circ         | test_ood_ecc  | state     | ridge    | log_v0         |     -0.268892    |
|     44 | M3_circ         | test_id_ellip | state     | ridge    | log_a          |     -4.44591     |
|     44 | M3_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -1.45002     |
|     44 | M3_circ         | test_id_ellip | state     | ridge    | e              |     -0.390644    |
|     44 | M3_circ         | test_ood_ecc  | state     | ridge    | e              |   -103.231       |
|     44 | M3_circ         | test_id_ellip | state     | ridge    | log_h          |     -3.23387     |
|     44 | M3_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -7.06574     |
|     44 | M3_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -4.44591     |
|     44 | M3_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.45002     |
|     44 | M3_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.813299    |
|     44 | M3_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.88197     |
|     44 | M3_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.53713     |
|     44 | M3_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.620074    |
|     44 | M3_circ         | test_id_ellip | state     | poly2    | log_a          |     -0.0490819   |
|     44 | M3_circ         | test_ood_ecc  | state     | poly2    | log_a          |      0.198176    |
|     44 | M3_circ         | test_id_ellip | state     | poly2    | e              |     -1.80339     |
|     44 | M3_circ         | test_ood_ecc  | state     | poly2    | e              |    -36.7275      |
|     44 | M3_circ         | test_id_ellip | state     | poly2    | log_h          |     -0.683261    |
|     44 | M3_circ         | test_ood_ecc  | state     | poly2    | log_h          |     -1.41762     |
|     44 | M3_circ         | test_id_ellip | state     | poly2    | log_abs_energy |     -0.0490819   |
|     44 | M3_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.198176    |
|     44 | M3_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.673939    |
|     44 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.539948    |
|     44 | M3_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.603501    |
|     44 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.402368    |
|     44 | M3_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.246057    |
|     44 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |     -0.681911    |
|     44 | M3_circ         | test_id_ellip | log_omega | ridge    | e              |      0.330562    |
|     44 | M3_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -80.1168      |
|     44 | M3_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.306965    |
|     44 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -4.55532     |
|     44 | M3_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.246057    |
|     44 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -0.681911    |
|     44 | M3_circ         | test_id_ellip | log_omega | poly2    | log_r0         |     -0.589371    |
|     44 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |     -0.361762    |
|     44 | M3_circ         | test_id_ellip | log_omega | poly2    | log_v0         |     -0.973165    |
|     44 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |     -0.455647    |
|     44 | M3_circ         | test_id_ellip | log_omega | poly2    | log_a          |     -1.90928     |
|     44 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |     -3.97912     |
|     44 | M3_circ         | test_id_ellip | log_omega | poly2    | e              |     -1.66521     |
|     44 | M3_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -82.5661      |
|     44 | M3_circ         | test_id_ellip | log_omega | poly2    | log_h          |     -2.80878     |
|     44 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -9.37176     |
|     44 | M3_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |     -1.90928     |
|     44 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -3.97912     |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_r0         |      0.457383    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_r0         |      0.0994171   |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_v0         |      0.175328    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.466032    |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_a          |      0.676327    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_a          |      0.728179    |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | e              |      0.00151859  |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | e              |   -114.323       |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_h          |      0.552631    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_h          |     -2.65073     |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_abs_energy |      0.676327    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.728179    |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_r0         |      0.429959    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.0907403   |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_v0         |      0.17429     |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.468866    |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_a          |      0.70912     |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_a          |      0.513232    |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | e              |      0.00404697  |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | e              |   -111.508       |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_h          |      0.589666    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_h          |     -2.40149     |
|     44 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_abs_energy |      0.70912     |
|     44 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.513232    |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_r0         |     -1.21016     |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_r0         |     -0.246838    |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_v0         |     -0.845887    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_v0         |      0.11184     |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_a          |     -1.19595     |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_a          |     -1.42589     |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | e              |      0.0918458   |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | e              |   -105.006       |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_h          |     -0.909727    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_h          |     -7.21301     |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_abs_energy |     -1.19595     |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.42589     |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_r0         |      0.991751    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_r0         |      0.978644    |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_v0         |      0.987409    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_v0         |      0.968153    |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_a          |      0.969086    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_a          |      0.951321    |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | e              |      0.500798    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | e              |    -55.7817      |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_h          |      0.953724    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_h          |      0.557811    |
|     44 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_abs_energy |      0.969086    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.951321    |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_r0         |      0.943126    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.814915    |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_v0         |      0.908587    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.753142    |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_a          |      0.901939    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_a          |      0.755652    |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | e              |      0.708154    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | e              |    -28.4824      |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_h          |      0.912873    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_h          |      0.149285    |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.901939    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.755652    |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_r0         |      0.978296    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.876589    |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_v0         |      0.979328    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.891719    |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_a          |      0.879871    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_a          |      0.513257    |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | e              |      0.638135    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | e              |    -38.8489      |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_h          |      0.947849    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_h          |      0.60529     |
|     44 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.879871    |
|     44 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.513257    |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_r0         |      0.433751    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.706732    |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_v0         |      0.0847948   |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.945694    |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_a          |      0.721759    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_a          |      0.23932     |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | e              |      0.000184966 |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | e              |   -106.089       |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_h          |      0.68255     |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_h          |     -1.50688     |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_abs_energy |      0.721759    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.23932     |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_r0         |      0.407171    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_r0         |     -1.08515     |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_v0         |      0.105112    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.82814     |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_a          |      0.708576    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_a          |     -0.56685     |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | e              |      0.00214289  |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | e              |    -95.1149      |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_h          |      0.682942    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_h          |     -1.57798     |
|     44 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_abs_energy |      0.708576    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -0.56685     |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_r0         |     -1.79187     |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_r0         |     -0.89417     |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_v0         |     -1.35113     |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_v0         |     -0.0407937   |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_a          |     -1.61871     |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_a          |     -2.07969     |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | e              |     -0.0718823   |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | e              |   -104.718       |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_h          |     -1.32191     |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_h          |     -8.52332     |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_abs_energy |     -1.61871     |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.07969     |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_r0         |      0.99789     |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_r0         |      0.995014    |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_v0         |      0.994318    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_v0         |      0.977147    |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_a          |      0.96739     |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_a          |      0.901572    |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | e              |     -0.0915165   |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | e              |   -148.206       |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_h          |      0.997435    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_h          |      0.896393    |
|     44 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_abs_energy |      0.96739     |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.901572    |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_r0         |      0.758641    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.362114    |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_v0         |      0.479393    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_v0         |     -0.0436494   |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_a          |      0.990345    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_a          |      0.926343    |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | e              |      0.48905     |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | e              |    -18.1807      |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_h          |      0.938069    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_h          |      0.207421    |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.990345    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.926343    |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_r0         |      0.503663    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_r0         |     -1.06165     |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_v0         |     -0.223093    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_v0         |     -1.82283     |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_a          |      0.987134    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_a          |      0.94631     |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | e              |     -0.751546    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | e              |    -64.7343      |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_h          |      0.880284    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_h          |      0.0825507   |
|     44 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.987134    |
|     44 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.94631     |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_r0         |      0.60892     |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.455693    |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_v0         |      0.236836    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.7894      |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_a          |      0.816672    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_a          |      0.439483    |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | e              |     -0.00294125  |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | e              |   -105.54        |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_h          |      0.750809    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_h          |     -1.62629     |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_abs_energy |      0.816672    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.439483    |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_r0         |      0.59041     |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.647876    |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_v0         |      0.237956    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.784024    |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_a          |      0.800165    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_a          |      0.137915    |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | e              |     -0.00996807  |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | e              |   -102.298       |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_h          |      0.741394    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_h          |     -1.6809      |
|     44 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_abs_energy |      0.800165    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.137915    |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_r0         |     -2.20495     |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_r0         |     -1.00355     |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_v0         |     -1.73762     |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_v0         |     -0.297357    |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_a          |     -1.9658      |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_a          |     -2.399       |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | e              |     -0.289894    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | e              |   -100.32        |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_h          |     -1.69182     |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_h          |     -8.71739     |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_abs_energy |     -1.9658      |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.399       |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_r0         |      0.99596     |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_r0         |      0.993258    |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_v0         |      0.997275    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_v0         |      0.977662    |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_a          |      0.987491    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_a          |      0.939843    |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | e              |      0.499636    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | e              |    -68.5616      |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_h          |      0.980945    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_h          |      0.868458    |
|     44 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_abs_energy |      0.987491    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.939843    |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_r0         |      0.748503    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.402791    |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_v0         |      0.506459    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.0105015   |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_a          |      0.99748     |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_a          |      0.944005    |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | e              |      0.663012    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | e              |    -24.6659      |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_h          |      0.976905    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_h          |      0.0734868   |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.99748     |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.944005    |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_r0         |     -2.30489     |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_r0         |     -1.45942     |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_v0         |     -7.27772     |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_v0         |     -2.35031     |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_a          |      0.994583    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_a          |      0.875823    |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | e              |     -5.51074     |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | e              |    -80.1959      |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_h          |      0.235813    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.204435    |
|     44 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.994583    |
|     44 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.875823    |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_r0         |     -1.09993     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.28753     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_v0         |     -0.693261    |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.0287097   |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_a          |     -1.33482     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_a          |     -1.33482     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | e              |     -0.000218993 |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | e              |   -110.341       |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_h          |     -1.0263      |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_h          |     -7.82839     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_abs_energy |     -1.33482     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -1.33482     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_r0         |     -1.09993     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.28753     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_v0         |     -0.693261    |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0287097   |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_a          |     -1.33482     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_a          |     -1.33482     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | e              |     -0.000218993 |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | e              |   -110.341       |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_h          |     -1.0263      |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_h          |     -7.82839     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_abs_energy |     -1.33482     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.33482     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_r0         |     -2.01645     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_r0         |     -0.907906    |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_v0         |     -1.44864     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_v0         |     -0.308462    |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_a          |     -2.27999     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_a          |     -1.97895     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | e              |     -0.296937    |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | e              |   -114.712       |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_h          |     -1.88489     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_h          |     -7.93038     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_abs_energy |     -2.27999     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.97895     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_r0         |      0.928294    |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_r0         |      0.870876    |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_v0         |      0.87903     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_v0         |      0.903364    |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_a          |      0.488186    |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_a          |      0.317476    |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | e              |      0.544185    |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | e              |    -23.0536      |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_h          |      0.56596     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_h          |     -1.84971     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_abs_energy |      0.488186    |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.317476    |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_r0         |     -1.3806      |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_r0         |     -0.494468    |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_v0         |     -1.72741     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_v0         |     -0.298306    |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_a          |     -1.01556     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_a          |     -0.93583     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | e              |     -1.04062     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | e              |   -112.82        |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_h          |     -0.970199    |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_h          |     -6.63044     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_abs_energy |     -1.01556     |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -0.93583     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_r0         |  -1639.02        |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_r0         |     -7.06675     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_v0         |    -84.0164      |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_v0         |     -3.92786     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_a          |  -5990.73        |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_a          |     -6.61709     |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | e              | -10767.8         |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | e              |   -117.125       |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_h          | -12516.1         |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_h          |    -10.5904      |
|     44 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_abs_energy |  -5990.73        |
|     44 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -6.61709     |
|     45 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_r0         |      0.587848    |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_r0         |      0.0349743   |
|     45 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_v0         |      0.34802     |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_v0         |      0.0473316   |
|     45 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_a          |      0.625846    |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_a          |     -0.104624    |
|     45 | M0_Koopman      | test_id_ellip | norm      | ridge    | e              |      0.0123385   |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | e              |   -114.974       |
|     45 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_h          |      0.457775    |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_h          |     -4.36316     |
|     45 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_abs_energy |      0.625846    |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -0.104624    |
|     45 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_r0         |      0.588814    |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_r0         |      0.0136566   |
|     45 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_v0         |      0.35281     |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_v0         |      0.0716032   |
|     45 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_a          |      0.679518    |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_a          |     -0.32405     |
|     45 | M0_Koopman      | test_id_ellip | norm      | poly2    | e              |      0.0331446   |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | e              |   -114.147       |
|     45 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_h          |      0.528897    |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_h          |     -4.50216     |
|     45 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_abs_energy |      0.679518    |
|     45 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -0.32405     |
|     45 | M0_Koopman      | test_id_ellip | state     | ridge    | log_r0         |     -1.28932     |
|     45 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_r0         |     -4.53594     |
|     45 | M0_Koopman      | test_id_ellip | state     | ridge    | log_v0         |     -0.594721    |
|     45 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_v0         |     -2.44425     |
|     45 | M0_Koopman      | test_id_ellip | state     | ridge    | log_a          |     -1.40637     |
|     45 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_a          |     -2.94683     |
|     45 | M0_Koopman      | test_id_ellip | state     | ridge    | e              |     -0.0304543   |
|     45 | M0_Koopman      | test_ood_ecc  | state     | ridge    | e              |   -176.579       |
|     45 | M0_Koopman      | test_id_ellip | state     | ridge    | log_h          |     -1.10383     |
|     45 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_h          |    -12.0962      |
|     45 | M0_Koopman      | test_id_ellip | state     | ridge    | log_abs_energy |     -1.40637     |
|     45 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.94683     |
|     45 | M0_Koopman      | test_id_ellip | state     | poly2    | log_r0         |      0.99939     |
|     45 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_r0         |      0.990599    |
|     45 | M0_Koopman      | test_id_ellip | state     | poly2    | log_v0         |      0.993491    |
|     45 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_v0         |      0.929795    |
|     45 | M0_Koopman      | test_id_ellip | state     | poly2    | log_a          |      0.987827    |
|     45 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_a          |      0.504375    |
|     45 | M0_Koopman      | test_id_ellip | state     | poly2    | e              |      0.781782    |
|     45 | M0_Koopman      | test_ood_ecc  | state     | poly2    | e              |    -71.4513      |
|     45 | M0_Koopman      | test_id_ellip | state     | poly2    | log_h          |      0.990622    |
|     45 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_h          |     -0.627592    |
|     45 | M0_Koopman      | test_id_ellip | state     | poly2    | log_abs_energy |      0.987827    |
|     45 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.504375    |
|     45 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_r0         |      0.969632    |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.913851    |
|     45 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_v0         |      0.899647    |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.708854    |
|     45 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_a          |      0.888813    |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_a          |     -1.81414     |
|     45 | M0_Koopman      | test_id_ellip | log_omega | ridge    | e              |      0.0303094   |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | e              |   -137.243       |
|     45 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_h          |      0.82524     |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_h          |     -5.17771     |
|     45 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.888813    |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -1.81414     |
|     45 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_r0         |      0.985818    |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.923716    |
|     45 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_v0         |      0.546534    |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.238048    |
|     45 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_a          |     -0.0577763   |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_a          |     -6.64613     |
|     45 | M0_Koopman      | test_id_ellip | log_omega | poly2    | e              |    -84.3819      |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | e              |  -5046.19        |
|     45 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_h          |     -1.18387     |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_h          |    -17.0529      |
|     45 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_abs_energy |     -0.0577763   |
|     45 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -6.64613     |
|     45 | M1_circ         | test_id_ellip | norm      | ridge    | log_r0         |      0.602648    |
|     45 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |      0.0208382   |
|     45 | M1_circ         | test_id_ellip | norm      | ridge    | log_v0         |      0.367677    |
|     45 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |      0.0415306   |
|     45 | M1_circ         | test_id_ellip | norm      | ridge    | log_a          |      0.616204    |
|     45 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_a          |     -0.0972866   |
|     45 | M1_circ         | test_id_ellip | norm      | ridge    | e              |      0.01527     |
|     45 | M1_circ         | test_ood_ecc  | norm      | ridge    | e              |   -115.223       |
|     45 | M1_circ         | test_id_ellip | norm      | ridge    | log_h          |      0.442587    |
|     45 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -4.3395      |
|     45 | M1_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |      0.616204    |
|     45 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -0.0972866   |
|     45 | M1_circ         | test_id_ellip | norm      | poly2    | log_r0         |      0.604684    |
|     45 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.00368819  |
|     45 | M1_circ         | test_id_ellip | norm      | poly2    | log_v0         |      0.367881    |
|     45 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |      0.0703339   |
|     45 | M1_circ         | test_id_ellip | norm      | poly2    | log_a          |      0.669769    |
|     45 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_a          |     -0.331652    |
|     45 | M1_circ         | test_id_ellip | norm      | poly2    | e              |      0.0321874   |
|     45 | M1_circ         | test_ood_ecc  | norm      | poly2    | e              |   -114.104       |
|     45 | M1_circ         | test_id_ellip | norm      | poly2    | log_h          |      0.510672    |
|     45 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -4.45913     |
|     45 | M1_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |      0.669769    |
|     45 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -0.331652    |
|     45 | M1_circ         | test_id_ellip | state     | ridge    | log_r0         |     -1.20049     |
|     45 | M1_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -6.75948     |
|     45 | M1_circ         | test_id_ellip | state     | ridge    | log_v0         |     -0.556894    |
|     45 | M1_circ         | test_ood_ecc  | state     | ridge    | log_v0         |     -3.17553     |
|     45 | M1_circ         | test_id_ellip | state     | ridge    | log_a          |     -1.29917     |
|     45 | M1_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -4.49197     |
|     45 | M1_circ         | test_id_ellip | state     | ridge    | e              |     -0.0575842   |
|     45 | M1_circ         | test_ood_ecc  | state     | ridge    | e              |   -228.971       |
|     45 | M1_circ         | test_id_ellip | state     | ridge    | log_h          |     -1.04537     |
|     45 | M1_circ         | test_ood_ecc  | state     | ridge    | log_h          |    -16.1019      |
|     45 | M1_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -1.29917     |
|     45 | M1_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -4.49197     |
|     45 | M1_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.999651    |
|     45 | M1_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.966337    |
|     45 | M1_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.993967    |
|     45 | M1_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.965009    |
|     45 | M1_circ         | test_id_ellip | state     | poly2    | log_a          |      0.989411    |
|     45 | M1_circ         | test_ood_ecc  | state     | poly2    | log_a          |     -2.16325     |
|     45 | M1_circ         | test_id_ellip | state     | poly2    | e              |      0.800003    |
|     45 | M1_circ         | test_ood_ecc  | state     | poly2    | e              |    -95.3872      |
|     45 | M1_circ         | test_id_ellip | state     | poly2    | log_h          |      0.991908    |
|     45 | M1_circ         | test_ood_ecc  | state     | poly2    | log_h          |      0.341448    |
|     45 | M1_circ         | test_id_ellip | state     | poly2    | log_abs_energy |      0.989411    |
|     45 | M1_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |     -2.16325     |
|     45 | M1_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.988507    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.953151    |
|     45 | M1_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.991259    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.967723    |
|     45 | M1_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.915517    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |      0.513414    |
|     45 | M1_circ         | test_id_ellip | log_omega | ridge    | e              |      0.511464    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -22.5529      |
|     45 | M1_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.899511    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -0.649459    |
|     45 | M1_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.915517    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.513414    |
|     45 | M1_circ         | test_id_ellip | log_omega | poly2    | log_r0         |      0.995379    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.976225    |
|     45 | M1_circ         | test_id_ellip | log_omega | poly2    | log_v0         |      0.998253    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.960846    |
|     45 | M1_circ         | test_id_ellip | log_omega | poly2    | log_a          |      0.963994    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |      0.152593    |
|     45 | M1_circ         | test_id_ellip | log_omega | poly2    | e              |      0.713953    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -21.6153      |
|     45 | M1_circ         | test_id_ellip | log_omega | poly2    | log_h          |      0.968566    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.18145     |
|     45 | M1_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.963994    |
|     45 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.152593    |
|     45 | M2_circ         | test_id_ellip | norm      | ridge    | log_r0         |      0.592333    |
|     45 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |      0.0161935   |
|     45 | M2_circ         | test_id_ellip | norm      | ridge    | log_v0         |      0.375779    |
|     45 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |      0.0636966   |
|     45 | M2_circ         | test_id_ellip | norm      | ridge    | log_a          |      0.584144    |
|     45 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_a          |     -0.180081    |
|     45 | M2_circ         | test_id_ellip | norm      | ridge    | e              |      0.0204252   |
|     45 | M2_circ         | test_ood_ecc  | norm      | ridge    | e              |   -114.79        |
|     45 | M2_circ         | test_id_ellip | norm      | ridge    | log_h          |      0.403249    |
|     45 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -4.68079     |
|     45 | M2_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |      0.584144    |
|     45 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -0.180081    |
|     45 | M2_circ         | test_id_ellip | norm      | poly2    | log_r0         |      0.593297    |
|     45 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |      0.00494239  |
|     45 | M2_circ         | test_id_ellip | norm      | poly2    | log_v0         |      0.375443    |
|     45 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |      0.0863297   |
|     45 | M2_circ         | test_id_ellip | norm      | poly2    | log_a          |      0.62939     |
|     45 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_a          |     -0.382359    |
|     45 | M2_circ         | test_id_ellip | norm      | poly2    | e              |      0.0328455   |
|     45 | M2_circ         | test_ood_ecc  | norm      | poly2    | e              |   -114.971       |
|     45 | M2_circ         | test_id_ellip | norm      | poly2    | log_h          |      0.459707    |
|     45 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -4.89582     |
|     45 | M2_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |      0.62939     |
|     45 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -0.382359    |
|     45 | M2_circ         | test_id_ellip | state     | ridge    | log_r0         |     -1.34465     |
|     45 | M2_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -2.18395     |
|     45 | M2_circ         | test_id_ellip | state     | ridge    | log_v0         |     -0.694203    |
|     45 | M2_circ         | test_ood_ecc  | state     | ridge    | log_v0         |     -1.17738     |
|     45 | M2_circ         | test_id_ellip | state     | ridge    | log_a          |     -1.33267     |
|     45 | M2_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -2.15232     |
|     45 | M2_circ         | test_id_ellip | state     | ridge    | e              |     -0.188994    |
|     45 | M2_circ         | test_ood_ecc  | state     | ridge    | e              |   -151.374       |
|     45 | M2_circ         | test_id_ellip | state     | ridge    | log_h          |     -1.15944     |
|     45 | M2_circ         | test_ood_ecc  | state     | ridge    | log_h          |    -10.0799      |
|     45 | M2_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -1.33267     |
|     45 | M2_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.15232     |
|     45 | M2_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.999341    |
|     45 | M2_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.94391     |
|     45 | M2_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.995342    |
|     45 | M2_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.734858    |
|     45 | M2_circ         | test_id_ellip | state     | poly2    | log_a          |      0.986343    |
|     45 | M2_circ         | test_ood_ecc  | state     | poly2    | log_a          |      0.749498    |
|     45 | M2_circ         | test_id_ellip | state     | poly2    | e              |      0.708665    |
|     45 | M2_circ         | test_ood_ecc  | state     | poly2    | e              |   -137.74        |
|     45 | M2_circ         | test_id_ellip | state     | poly2    | log_h          |      0.992089    |
|     45 | M2_circ         | test_ood_ecc  | state     | poly2    | log_h          |     -0.613836    |
|     45 | M2_circ         | test_id_ellip | state     | poly2    | log_abs_energy |      0.986343    |
|     45 | M2_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.749498    |
|     45 | M2_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.966062    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.932772    |
|     45 | M2_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.956436    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.911282    |
|     45 | M2_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.766154    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |      0.254168    |
|     45 | M2_circ         | test_id_ellip | log_omega | ridge    | e              |      0.184654    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -15.4645      |
|     45 | M2_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.625956    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -1.24515     |
|     45 | M2_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.766154    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.254168    |
|     45 | M2_circ         | test_id_ellip | log_omega | poly2    | log_r0         |      0.993347    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.976034    |
|     45 | M2_circ         | test_id_ellip | log_omega | poly2    | log_v0         |      0.987782    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.911949    |
|     45 | M2_circ         | test_id_ellip | log_omega | poly2    | log_a          |      0.938384    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |      0.0468909   |
|     45 | M2_circ         | test_id_ellip | log_omega | poly2    | e              |      0.783923    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -19.0353      |
|     45 | M2_circ         | test_id_ellip | log_omega | poly2    | log_h          |      0.929266    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.834885    |
|     45 | M2_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.938384    |
|     45 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.0468909   |
|     45 | M3_circ         | test_id_ellip | norm      | ridge    | log_r0         |     -1.09788     |
|     45 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.419522    |
|     45 | M3_circ         | test_id_ellip | norm      | ridge    | log_v0         |     -0.506172    |
|     45 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.00243391  |
|     45 | M3_circ         | test_id_ellip | norm      | ridge    | log_a          |     -1.33482     |
|     45 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_a          |     -1.33482     |
|     45 | M3_circ         | test_id_ellip | norm      | ridge    | e              |     -0.00543215  |
|     45 | M3_circ         | test_ood_ecc  | norm      | ridge    | e              |   -105.167       |
|     45 | M3_circ         | test_id_ellip | norm      | ridge    | log_h          |     -0.998735    |
|     45 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -7.75128     |
|     45 | M3_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |     -1.33482     |
|     45 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -1.33482     |
|     45 | M3_circ         | test_id_ellip | norm      | poly2    | log_r0         |     -1.09788     |
|     45 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.419522    |
|     45 | M3_circ         | test_id_ellip | norm      | poly2    | log_v0         |     -0.506172    |
|     45 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.00243391  |
|     45 | M3_circ         | test_id_ellip | norm      | poly2    | log_a          |     -1.33482     |
|     45 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_a          |     -1.33482     |
|     45 | M3_circ         | test_id_ellip | norm      | poly2    | e              |     -0.00543215  |
|     45 | M3_circ         | test_ood_ecc  | norm      | poly2    | e              |   -105.167       |
|     45 | M3_circ         | test_id_ellip | norm      | poly2    | log_h          |     -0.998735    |
|     45 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -7.75128     |
|     45 | M3_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |     -1.33482     |
|     45 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.33482     |
|     45 | M3_circ         | test_id_ellip | state     | ridge    | log_r0         |     -5.00628     |
|     45 | M3_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -0.818996    |
|     45 | M3_circ         | test_id_ellip | state     | ridge    | log_v0         |     -2.70862     |
|     45 | M3_circ         | test_ood_ecc  | state     | ridge    | log_v0         |      0.000198132 |
|     45 | M3_circ         | test_id_ellip | state     | ridge    | log_a          |     -5.47496     |
|     45 | M3_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -3.29939     |
|     45 | M3_circ         | test_id_ellip | state     | ridge    | e              |     -0.417417    |
|     45 | M3_circ         | test_ood_ecc  | state     | ridge    | e              |   -111.417       |
|     45 | M3_circ         | test_id_ellip | state     | ridge    | log_h          |     -4.31474     |
|     45 | M3_circ         | test_ood_ecc  | state     | ridge    | log_h          |    -10.0455      |
|     45 | M3_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -5.47496     |
|     45 | M3_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -3.29939     |
|     45 | M3_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.580459    |
|     45 | M3_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.748392    |
|     45 | M3_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.770752    |
|     45 | M3_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.841092    |
|     45 | M3_circ         | test_id_ellip | state     | poly2    | log_a          |     -0.117268    |
|     45 | M3_circ         | test_ood_ecc  | state     | poly2    | log_a          |     -0.365849    |
|     45 | M3_circ         | test_id_ellip | state     | poly2    | e              |      0.105942    |
|     45 | M3_circ         | test_ood_ecc  | state     | poly2    | e              |    -75.63        |
|     45 | M3_circ         | test_id_ellip | state     | poly2    | log_h          |     -0.175836    |
|     45 | M3_circ         | test_ood_ecc  | state     | poly2    | log_h          |     -3.83068     |
|     45 | M3_circ         | test_id_ellip | state     | poly2    | log_abs_energy |     -0.117268    |
|     45 | M3_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.365849    |
|     45 | M3_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.259857    |
|     45 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.691564    |
|     45 | M3_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.426971    |
|     45 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.516086    |
|     45 | M3_circ         | test_id_ellip | log_omega | ridge    | log_a          |     -0.123527    |
|     45 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |     -0.430289    |
|     45 | M3_circ         | test_id_ellip | log_omega | ridge    | e              |      0.144018    |
|     45 | M3_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -64.9289      |
|     45 | M3_circ         | test_id_ellip | log_omega | ridge    | log_h          |     -0.242027    |
|     45 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -3.72602     |
|     45 | M3_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |     -0.123527    |
|     45 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -0.430289    |
|     45 | M3_circ         | test_id_ellip | log_omega | poly2    | log_r0         |     -1.90937     |
|     45 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.571379    |
|     45 | M3_circ         | test_id_ellip | log_omega | poly2    | log_v0         |     -0.148344    |
|     45 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.468949    |
|     45 | M3_circ         | test_id_ellip | log_omega | poly2    | log_a          |     -4.80288     |
|     45 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |     -0.879701    |
|     45 | M3_circ         | test_id_ellip | log_omega | poly2    | e              |     -1.21677     |
|     45 | M3_circ         | test_ood_ecc  | log_omega | poly2    | e              |   -117.149       |
|     45 | M3_circ         | test_id_ellip | log_omega | poly2    | log_h          |     -4.30405     |
|     45 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -4.5956      |
|     45 | M3_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |     -4.80288     |
|     45 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -0.879701    |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_r0         |      0.715909    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.049483    |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_v0         |      0.377172    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.146146    |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_a          |      0.771871    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_a          |      0.237454    |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | e              |      0.00742484  |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | e              |   -113.519       |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_h          |      0.614532    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_h          |     -3.12654     |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_abs_energy |      0.771871    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.237454    |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_r0         |      0.72177     |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.286556    |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_v0         |      0.376469    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.167395    |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_a          |      0.844067    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_a          |     -0.230029    |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | e              |      0.0110245   |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | e              |   -109.142       |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_h          |      0.693992    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_h          |     -3.08092     |
|     45 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_abs_energy |      0.844067    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -0.230029    |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_r0         |     -1.56165     |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_r0         |     -2.48195     |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_v0         |     -0.79535     |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_v0         |     -1.16933     |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_a          |     -1.63686     |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_a          |     -2.39625     |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | e              |     -0.0723053   |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | e              |   -189.581       |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_h          |     -1.38039     |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_h          |    -11.8638      |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_abs_energy |     -1.63686     |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.39625     |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_r0         |      0.999155    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_r0         |      0.957328    |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_v0         |      0.996192    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_v0         |      0.975989    |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_a          |      0.993435    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_a          |      0.292426    |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | e              |      0.806699    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | e              |   -178.605       |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_h          |      0.988595    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_h          |     -0.134101    |
|     45 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_abs_energy |      0.993435    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.292426    |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_r0         |      0.969052    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.863388    |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_v0         |      0.912512    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.797516    |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_a          |      0.956168    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_a          |      0.159579    |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | e              |      0.49986     |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | e              |    -46.7595      |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_h          |      0.925763    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_h          |     -1.05012     |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.956168    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.159579    |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_r0         |      0.989766    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.856598    |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_v0         |      0.986911    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.905121    |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_a          |      0.964991    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_a          |     -0.696563    |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | e              |      0.727452    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | e              |   -143.603       |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_h          |      0.969334    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_h          |      0.103961    |
|     45 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.964991    |
|     45 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -0.696563    |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_r0         |      0.291653    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.613964    |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_v0         |     -0.0839839   |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.502376    |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_a          |      0.853108    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_a          |     -0.12925     |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | e              |     -0.00762184  |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | e              |   -105.965       |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_h          |      0.908222    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_h          |     -2.48342     |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_abs_energy |      0.853108    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -0.12925     |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_r0         |      0.303704    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.580014    |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_v0         |      0.0522526   |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.25988     |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_a          |      0.798743    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_a          |     -0.622906    |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | e              |      0.098704    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | e              |    -98.5462      |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_h          |      0.875721    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_h          |     -2.84995     |
|     45 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_abs_energy |      0.798743    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -0.622906    |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_r0         |     -1.84582     |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_r0         |    -28.7401      |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_v0         |     -0.953776    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_v0         |    -16.2546      |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_a          |     -1.99971     |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_a          |    -15.109       |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | e              |     -0.409839    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | e              |   -168.974       |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_h          |     -1.53239     |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_h          |    -27.3838      |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_abs_energy |     -1.99971     |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_abs_energy |    -15.109       |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_r0         |      0.999806    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_r0         |      0.983776    |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_v0         |      0.998023    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_v0         |      0.94846     |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_a          |      0.994297    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_a          |     -0.224407    |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | e              |      0.861801    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | e              |   -301.363       |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_h          |      0.995469    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_h          |     -0.224407    |
|     45 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_abs_energy |      0.994297    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.224407    |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_r0         |      0.705673    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.437014    |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_v0         |      0.472412    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.414353    |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_a          |      0.990399    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_a          |      0.621683    |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | e              |      0.798271    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | e              |    -33.789       |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_h          |      0.978376    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_h          |      0.0446727   |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.990399    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.621683    |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_r0         |      0.357951    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_r0         |    -56.3575      |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_v0         |     -0.282418    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_v0         |    -74.9608      |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_a          |      0.988536    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_a          |      0.229211    |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | e              |      0.598041    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | e              |   -649.733       |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_h          |      0.859868    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_h          |     -4.41517     |
|     45 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.988536    |
|     45 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.229211    |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_r0         |      0.234816    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.543408    |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_v0         |     -0.13256     |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.45596     |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_a          |      0.838331    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_a          |     -0.0590358   |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | e              |     -0.0022591   |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | e              |   -104.283       |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_h          |      0.949022    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_h          |     -2.28591     |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_abs_energy |      0.838331    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -0.0590358   |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_r0         |      0.32225     |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.410722    |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_v0         |      0.0474561   |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.27045     |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_a          |      0.807195    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_a          |     -0.158519    |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | e              |      0.108918    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | e              |   -102.362       |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_h          |      0.904262    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_h          |     -2.46097     |
|     45 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_abs_energy |      0.807195    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -0.158519    |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_r0         |     -2.02012     |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_r0         |    -42.3037      |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_v0         |     -0.982217    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_v0         |    -22.3981      |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_a          |     -2.11485     |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_a          |    -24.2135      |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | e              |     -0.208711    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | e              |   -125.312       |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_h          |     -1.66048     |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_h          |    -34.9233      |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_abs_energy |     -2.11485     |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_abs_energy |    -24.2135      |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_r0         |      0.999885    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_r0         |      0.998481    |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_v0         |      0.998585    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_v0         |      0.891091    |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_a          |      0.987372    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_a          |     -1.15757     |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | e              |      0.442097    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | e              | -15239           |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_h          |      0.995936    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_h          |     -0.0186704   |
|     45 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_abs_energy |      0.987372    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_abs_energy |     -1.15757     |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_r0         |      0.785792    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.0958927   |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_v0         |      0.558109    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_v0         |     -0.084108    |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_a          |      0.992389    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_a          |      0.787019    |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | e              |      0.506524    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | e              |    -23.6258      |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_h          |      0.930097    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_h          |     -0.693348    |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.992389    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.787019    |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_r0         |      0.515031    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_r0         |     -1.21827     |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_v0         |      0.212473    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_v0         |     -1.2481      |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_a          |      0.995818    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_a          |      0.182803    |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | e              |     -0.299221    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | e              |   -361.283       |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_h          |      0.914807    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.669771    |
|     45 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.995818    |
|     45 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.182803    |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_r0         |     -1.09788     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.419522    |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_v0         |     -0.506172    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.00243391  |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_a          |     -1.33482     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_a          |     -1.33482     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | e              |     -0.00543215  |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | e              |   -105.167       |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_h          |     -0.998735    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_h          |     -7.75128     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_abs_energy |     -1.33482     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -1.33482     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_r0         |     -1.09788     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.419522    |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_v0         |     -0.506172    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.00243391  |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_a          |     -1.33482     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_a          |     -1.33482     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | e              |     -0.00543215  |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | e              |   -105.167       |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_h          |     -0.998735    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_h          |     -7.75128     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_abs_energy |     -1.33482     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.33482     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_r0         |     -2.87903     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_r0         |     -1.39614     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_v0         |     -1.26176     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_v0         |     -0.246133    |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_a          |     -3.69619     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_a          |     -3.92376     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | e              |     -0.473875    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | e              |   -125.704       |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_h          |     -3.03742     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_h          |    -12.3199      |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_abs_energy |     -3.69619     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_abs_energy |     -3.92376     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_r0         |      0.906212    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_r0         |      0.837099    |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_v0         |      0.872044    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_v0         |      0.860643    |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_a          |      0.618959    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_a          |     -0.0432557   |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | e              |      0.0609619   |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | e              |    -43.2428      |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_h          |      0.536277    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_h          |     -2.86701     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_abs_energy |      0.618959    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.0432557   |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_r0         |     -0.717155    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_r0         |     -1.77535     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_v0         |     -0.227337    |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_v0         |     -0.354623    |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_a          |     -1.40993     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_a          |     -3.72608     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | e              |     -0.28903     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | e              |   -101.645       |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_h          |     -1.36225     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_h          |    -10.3478      |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_abs_energy |     -1.40993     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -3.72608     |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_r0         |    -11.1004      |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_r0         |    -69.4951      |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_v0         |     -5.30514     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_v0         |    -22.4443      |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_a          |    -12.0588      |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_a          |   -119.681       |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | e              |     -5.50373     |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | e              |   -345.666       |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_h          |    -12.9681      |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_h          |   -130.709       |
|     45 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_abs_energy |    -12.0588      |
|     45 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_abs_energy |   -119.681       |
|     46 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_r0         |     -0.188183    |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_r0         |      0.385319    |
|     46 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_v0         |     -0.226949    |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.146281    |
|     46 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_a          |      0.0575295   |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_a          |      0.483657    |
|     46 | M0_Koopman      | test_id_ellip | norm      | ridge    | e              |      0.0262142   |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | e              |   -129.29        |
|     46 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_h          |     -0.138153    |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_h          |     -4.34071     |
|     46 | M0_Koopman      | test_id_ellip | norm      | ridge    | log_abs_energy |      0.0575295   |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.483657    |
|     46 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_r0         |     -0.211923    |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_r0         |      0.367547    |
|     46 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_v0         |     -0.144805    |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.112644    |
|     46 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_a          |     -0.0578772   |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_a          |      0.380501    |
|     46 | M0_Koopman      | test_id_ellip | norm      | poly2    | e              |     -0.0362172   |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | e              |   -128.003       |
|     46 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_h          |     -0.303099    |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_h          |     -4.34373     |
|     46 | M0_Koopman      | test_id_ellip | norm      | poly2    | log_abs_energy |     -0.0578772   |
|     46 | M0_Koopman      | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.380501    |
|     46 | M0_Koopman      | test_id_ellip | state     | ridge    | log_r0         |     -0.947203    |
|     46 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_r0         |     -0.101086    |
|     46 | M0_Koopman      | test_id_ellip | state     | ridge    | log_v0         |     -0.755805    |
|     46 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_v0         |      0.0282208   |
|     46 | M0_Koopman      | test_id_ellip | state     | ridge    | log_a          |     -1.44908     |
|     46 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_a          |     -1.26093     |
|     46 | M0_Koopman      | test_id_ellip | state     | ridge    | e              |     -0.217779    |
|     46 | M0_Koopman      | test_ood_ecc  | state     | ridge    | e              |   -141.283       |
|     46 | M0_Koopman      | test_id_ellip | state     | ridge    | log_h          |     -1.21728     |
|     46 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_h          |     -8.14892     |
|     46 | M0_Koopman      | test_id_ellip | state     | ridge    | log_abs_energy |     -1.44908     |
|     46 | M0_Koopman      | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.26093     |
|     46 | M0_Koopman      | test_id_ellip | state     | poly2    | log_r0         |      0.99973     |
|     46 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_r0         |      0.998194    |
|     46 | M0_Koopman      | test_id_ellip | state     | poly2    | log_v0         |      0.96419     |
|     46 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_v0         |      0.926756    |
|     46 | M0_Koopman      | test_id_ellip | state     | poly2    | log_a          |      0.564653    |
|     46 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_a          |      0.443425    |
|     46 | M0_Koopman      | test_id_ellip | state     | poly2    | e              |     -8.39466     |
|     46 | M0_Koopman      | test_ood_ecc  | state     | poly2    | e              |   -263.8         |
|     46 | M0_Koopman      | test_id_ellip | state     | poly2    | log_h          |      0.914167    |
|     46 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_h          |      0.676711    |
|     46 | M0_Koopman      | test_id_ellip | state     | poly2    | log_abs_energy |      0.564653    |
|     46 | M0_Koopman      | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.443425    |
|     46 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_r0         |      0.847717    |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.858657    |
|     46 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_v0         |      0.863644    |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.661647    |
|     46 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_a          |      0.557496    |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_a          |      0.262428    |
|     46 | M0_Koopman      | test_id_ellip | log_omega | ridge    | e              |      0.0645195   |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | e              |    -99.0829      |
|     46 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_h          |      0.479075    |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_h          |     -1.05363     |
|     46 | M0_Koopman      | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.557496    |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.262428    |
|     46 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_r0         |      0.860126    |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.951065    |
|     46 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_v0         |      0.859381    |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.794692    |
|     46 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_a          |     -0.155225    |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_a          |     -0.99882     |
|     46 | M0_Koopman      | test_id_ellip | log_omega | poly2    | e              |     -5.83015     |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | e              |   -206.53        |
|     46 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_h          |      0.732827    |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.6372      |
|     46 | M0_Koopman      | test_id_ellip | log_omega | poly2    | log_abs_energy |     -0.155225    |
|     46 | M0_Koopman      | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -0.99882     |
|     46 | M1_circ         | test_id_ellip | norm      | ridge    | log_r0         |     -0.155646    |
|     46 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |      0.419847    |
|     46 | M1_circ         | test_id_ellip | norm      | ridge    | log_v0         |     -0.203275    |
|     46 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.126675    |
|     46 | M1_circ         | test_id_ellip | norm      | ridge    | log_a          |      0.094558    |
|     46 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_a          |      0.499079    |
|     46 | M1_circ         | test_id_ellip | norm      | ridge    | e              |      0.0230527   |
|     46 | M1_circ         | test_ood_ecc  | norm      | ridge    | e              |   -128.979       |
|     46 | M1_circ         | test_id_ellip | norm      | ridge    | log_h          |     -0.10173     |
|     46 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -4.34107     |
|     46 | M1_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |      0.094558    |
|     46 | M1_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.499079    |
|     46 | M1_circ         | test_id_ellip | norm      | poly2    | log_r0         |     -0.188577    |
|     46 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |      0.397416    |
|     46 | M1_circ         | test_id_ellip | norm      | poly2    | log_v0         |     -0.131662    |
|     46 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.101237    |
|     46 | M1_circ         | test_id_ellip | norm      | poly2    | log_a          |     -0.0254932   |
|     46 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_a          |      0.397201    |
|     46 | M1_circ         | test_id_ellip | norm      | poly2    | e              |     -0.0371531   |
|     46 | M1_circ         | test_ood_ecc  | norm      | poly2    | e              |   -127.879       |
|     46 | M1_circ         | test_id_ellip | norm      | poly2    | log_h          |     -0.270433    |
|     46 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -4.34744     |
|     46 | M1_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |     -0.0254932   |
|     46 | M1_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.397201    |
|     46 | M1_circ         | test_id_ellip | state     | ridge    | log_r0         |     -1.24264     |
|     46 | M1_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -0.531852    |
|     46 | M1_circ         | test_id_ellip | state     | ridge    | log_v0         |     -1.00316     |
|     46 | M1_circ         | test_ood_ecc  | state     | ridge    | log_v0         |     -0.201084    |
|     46 | M1_circ         | test_id_ellip | state     | ridge    | log_a          |     -1.72143     |
|     46 | M1_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -1.51857     |
|     46 | M1_circ         | test_id_ellip | state     | ridge    | e              |     -0.154268    |
|     46 | M1_circ         | test_ood_ecc  | state     | ridge    | e              |   -136.975       |
|     46 | M1_circ         | test_id_ellip | state     | ridge    | log_h          |     -1.53267     |
|     46 | M1_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -8.67589     |
|     46 | M1_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -1.72143     |
|     46 | M1_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.51857     |
|     46 | M1_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.995379    |
|     46 | M1_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.998362    |
|     46 | M1_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.961024    |
|     46 | M1_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.929589    |
|     46 | M1_circ         | test_id_ellip | state     | poly2    | log_a          |      0.597146    |
|     46 | M1_circ         | test_ood_ecc  | state     | poly2    | log_a          |      0.40704     |
|     46 | M1_circ         | test_id_ellip | state     | poly2    | e              |     -5.5978      |
|     46 | M1_circ         | test_ood_ecc  | state     | poly2    | e              |   -276.854       |
|     46 | M1_circ         | test_id_ellip | state     | poly2    | log_h          |      0.911506    |
|     46 | M1_circ         | test_ood_ecc  | state     | poly2    | log_h          |      0.66176     |
|     46 | M1_circ         | test_id_ellip | state     | poly2    | log_abs_energy |      0.597146    |
|     46 | M1_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.40704     |
|     46 | M1_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.968593    |
|     46 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.959164    |
|     46 | M1_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.967578    |
|     46 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.917084    |
|     46 | M1_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.828823    |
|     46 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |      0.365089    |
|     46 | M1_circ         | test_id_ellip | log_omega | ridge    | e              |      0.388962    |
|     46 | M1_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -36.5762      |
|     46 | M1_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.845126    |
|     46 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -0.214923    |
|     46 | M1_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.828823    |
|     46 | M1_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.365089    |
|     46 | M1_circ         | test_id_ellip | log_omega | poly2    | log_r0         |      0.970453    |
|     46 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.9513      |
|     46 | M1_circ         | test_id_ellip | log_omega | poly2    | log_v0         |      0.962075    |
|     46 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.960299    |
|     46 | M1_circ         | test_id_ellip | log_omega | poly2    | log_a          |      0.435084    |
|     46 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |     -0.490849    |
|     46 | M1_circ         | test_id_ellip | log_omega | poly2    | e              |     -1.48533     |
|     46 | M1_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -60.4752      |
|     46 | M1_circ         | test_id_ellip | log_omega | poly2    | log_h          |      0.620879    |
|     46 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.407752    |
|     46 | M1_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.435084    |
|     46 | M1_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -0.490849    |
|     46 | M2_circ         | test_id_ellip | norm      | ridge    | log_r0         |     -0.177345    |
|     46 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |      0.42102     |
|     46 | M2_circ         | test_id_ellip | norm      | ridge    | log_v0         |     -0.218343    |
|     46 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.102623    |
|     46 | M2_circ         | test_id_ellip | norm      | ridge    | log_a          |      0.0651952   |
|     46 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_a          |      0.427724    |
|     46 | M2_circ         | test_id_ellip | norm      | ridge    | e              |      0.0265588   |
|     46 | M2_circ         | test_ood_ecc  | norm      | ridge    | e              |   -128.548       |
|     46 | M2_circ         | test_id_ellip | norm      | ridge    | log_h          |     -0.133132    |
|     46 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -4.60213     |
|     46 | M2_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |      0.0651952   |
|     46 | M2_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.427724    |
|     46 | M2_circ         | test_id_ellip | norm      | poly2    | log_r0         |     -0.208721    |
|     46 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |      0.401423    |
|     46 | M2_circ         | test_id_ellip | norm      | poly2    | log_v0         |     -0.142138    |
|     46 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0810011   |
|     46 | M2_circ         | test_id_ellip | norm      | poly2    | log_a          |     -0.0620652   |
|     46 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_a          |      0.316931    |
|     46 | M2_circ         | test_id_ellip | norm      | poly2    | e              |     -0.0289441   |
|     46 | M2_circ         | test_ood_ecc  | norm      | poly2    | e              |   -128.45        |
|     46 | M2_circ         | test_id_ellip | norm      | poly2    | log_h          |     -0.308111    |
|     46 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -4.70409     |
|     46 | M2_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |     -0.0620652   |
|     46 | M2_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.316931    |
|     46 | M2_circ         | test_id_ellip | state     | ridge    | log_r0         |     -1.20551     |
|     46 | M2_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -0.702659    |
|     46 | M2_circ         | test_id_ellip | state     | ridge    | log_v0         |     -1.05351     |
|     46 | M2_circ         | test_ood_ecc  | state     | ridge    | log_v0         |     -0.227007    |
|     46 | M2_circ         | test_id_ellip | state     | ridge    | log_a          |     -1.73958     |
|     46 | M2_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -1.66151     |
|     46 | M2_circ         | test_id_ellip | state     | ridge    | e              |     -0.0451099   |
|     46 | M2_circ         | test_ood_ecc  | state     | ridge    | e              |   -139.866       |
|     46 | M2_circ         | test_id_ellip | state     | ridge    | log_h          |     -1.45317     |
|     46 | M2_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -8.43883     |
|     46 | M2_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -1.73958     |
|     46 | M2_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.66151     |
|     46 | M2_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.991919    |
|     46 | M2_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.991241    |
|     46 | M2_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.967929    |
|     46 | M2_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.948863    |
|     46 | M2_circ         | test_id_ellip | state     | poly2    | log_a          |     -0.564499    |
|     46 | M2_circ         | test_ood_ecc  | state     | poly2    | log_a          |     -0.425765    |
|     46 | M2_circ         | test_id_ellip | state     | poly2    | e              |    -30.3807      |
|     46 | M2_circ         | test_ood_ecc  | state     | poly2    | e              |   -350.966       |
|     46 | M2_circ         | test_id_ellip | state     | poly2    | log_h          |      0.957369    |
|     46 | M2_circ         | test_ood_ecc  | state     | poly2    | log_h          |      0.737364    |
|     46 | M2_circ         | test_id_ellip | state     | poly2    | log_abs_energy |     -0.564499    |
|     46 | M2_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.425765    |
|     46 | M2_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.954644    |
|     46 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.845826    |
|     46 | M2_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.969513    |
|     46 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.808302    |
|     46 | M2_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.71967     |
|     46 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |     -0.0436113   |
|     46 | M2_circ         | test_id_ellip | log_omega | ridge    | e              |      0.4544      |
|     46 | M2_circ         | test_ood_ecc  | log_omega | ridge    | e              |   -105.8         |
|     46 | M2_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.699912    |
|     46 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -3.57599     |
|     46 | M2_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.71967     |
|     46 | M2_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -0.0436113   |
|     46 | M2_circ         | test_id_ellip | log_omega | poly2    | log_r0         |      0.973266    |
|     46 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.85837     |
|     46 | M2_circ         | test_id_ellip | log_omega | poly2    | log_v0         |      0.990386    |
|     46 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.89485     |
|     46 | M2_circ         | test_id_ellip | log_omega | poly2    | log_a          |      0.776514    |
|     46 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |      0.269663    |
|     46 | M2_circ         | test_id_ellip | log_omega | poly2    | e              |      0.493115    |
|     46 | M2_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -36.5295      |
|     46 | M2_circ         | test_id_ellip | log_omega | poly2    | log_h          |      0.738654    |
|     46 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -0.775887    |
|     46 | M2_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.776514    |
|     46 | M2_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.269663    |
|     46 | M3_circ         | test_id_ellip | norm      | ridge    | log_r0         |     -0.627001    |
|     46 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.215648    |
|     46 | M3_circ         | test_id_ellip | norm      | ridge    | log_v0         |     -0.314416    |
|     46 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.0539349   |
|     46 | M3_circ         | test_id_ellip | norm      | ridge    | log_a          |     -1.33482     |
|     46 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_a          |     -1.33482     |
|     46 | M3_circ         | test_id_ellip | norm      | ridge    | e              |     -0.00636694  |
|     46 | M3_circ         | test_ood_ecc  | norm      | ridge    | e              |   -118.909       |
|     46 | M3_circ         | test_id_ellip | norm      | ridge    | log_h          |     -1.1407      |
|     46 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_h          |     -8.14171     |
|     46 | M3_circ         | test_id_ellip | norm      | ridge    | log_abs_energy |     -1.33482     |
|     46 | M3_circ         | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -1.33482     |
|     46 | M3_circ         | test_id_ellip | norm      | poly2    | log_r0         |     -0.627001    |
|     46 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.215648    |
|     46 | M3_circ         | test_id_ellip | norm      | poly2    | log_v0         |     -0.314416    |
|     46 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0539349   |
|     46 | M3_circ         | test_id_ellip | norm      | poly2    | log_a          |     -1.33482     |
|     46 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_a          |     -1.33482     |
|     46 | M3_circ         | test_id_ellip | norm      | poly2    | e              |     -0.00636694  |
|     46 | M3_circ         | test_ood_ecc  | norm      | poly2    | e              |   -118.909       |
|     46 | M3_circ         | test_id_ellip | norm      | poly2    | log_h          |     -1.1407      |
|     46 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_h          |     -8.14171     |
|     46 | M3_circ         | test_id_ellip | norm      | poly2    | log_abs_energy |     -1.33482     |
|     46 | M3_circ         | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.33482     |
|     46 | M3_circ         | test_id_ellip | state     | ridge    | log_r0         |     -1.5619      |
|     46 | M3_circ         | test_ood_ecc  | state     | ridge    | log_r0         |     -0.650556    |
|     46 | M3_circ         | test_id_ellip | state     | ridge    | log_v0         |     -1.04796     |
|     46 | M3_circ         | test_ood_ecc  | state     | ridge    | log_v0         |     -0.198903    |
|     46 | M3_circ         | test_id_ellip | state     | ridge    | log_a          |     -3.54008     |
|     46 | M3_circ         | test_ood_ecc  | state     | ridge    | log_a          |     -2.59122     |
|     46 | M3_circ         | test_id_ellip | state     | ridge    | e              |      0.148793    |
|     46 | M3_circ         | test_ood_ecc  | state     | ridge    | e              |   -118.294       |
|     46 | M3_circ         | test_id_ellip | state     | ridge    | log_h          |     -2.74685     |
|     46 | M3_circ         | test_ood_ecc  | state     | ridge    | log_h          |     -8.91349     |
|     46 | M3_circ         | test_id_ellip | state     | ridge    | log_abs_energy |     -3.54008     |
|     46 | M3_circ         | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.59122     |
|     46 | M3_circ         | test_id_ellip | state     | poly2    | log_r0         |      0.519981    |
|     46 | M3_circ         | test_ood_ecc  | state     | poly2    | log_r0         |      0.730687    |
|     46 | M3_circ         | test_id_ellip | state     | poly2    | log_v0         |      0.459085    |
|     46 | M3_circ         | test_ood_ecc  | state     | poly2    | log_v0         |      0.691307    |
|     46 | M3_circ         | test_id_ellip | state     | poly2    | log_a          |      0.128613    |
|     46 | M3_circ         | test_ood_ecc  | state     | poly2    | log_a          |     -0.111446    |
|     46 | M3_circ         | test_id_ellip | state     | poly2    | e              |     -0.808286    |
|     46 | M3_circ         | test_ood_ecc  | state     | poly2    | e              |    -17.7278      |
|     46 | M3_circ         | test_id_ellip | state     | poly2    | log_h          |     -0.408561    |
|     46 | M3_circ         | test_ood_ecc  | state     | poly2    | log_h          |     -2.61828     |
|     46 | M3_circ         | test_id_ellip | state     | poly2    | log_abs_energy |      0.128613    |
|     46 | M3_circ         | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.111446    |
|     46 | M3_circ         | test_id_ellip | log_omega | ridge    | log_r0         |      0.687217    |
|     46 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.525934    |
|     46 | M3_circ         | test_id_ellip | log_omega | ridge    | log_v0         |      0.71336     |
|     46 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.224994    |
|     46 | M3_circ         | test_id_ellip | log_omega | ridge    | log_a          |      0.284002    |
|     46 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_a          |     -0.0336572   |
|     46 | M3_circ         | test_id_ellip | log_omega | ridge    | e              |      0.330439    |
|     46 | M3_circ         | test_ood_ecc  | log_omega | ridge    | e              |    -81.3733      |
|     46 | M3_circ         | test_id_ellip | log_omega | ridge    | log_h          |      0.134421    |
|     46 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_h          |     -3.75475     |
|     46 | M3_circ         | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.284002    |
|     46 | M3_circ         | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -0.0336572   |
|     46 | M3_circ         | test_id_ellip | log_omega | poly2    | log_r0         |     -0.500928    |
|     46 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.378719    |
|     46 | M3_circ         | test_id_ellip | log_omega | poly2    | log_v0         |     -1.61256     |
|     46 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.519641    |
|     46 | M3_circ         | test_id_ellip | log_omega | poly2    | log_a          |     -1.84339     |
|     46 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_a          |     -2.19488     |
|     46 | M3_circ         | test_id_ellip | log_omega | poly2    | e              |     -2.44851     |
|     46 | M3_circ         | test_ood_ecc  | log_omega | poly2    | e              |    -69.3142      |
|     46 | M3_circ         | test_id_ellip | log_omega | poly2    | log_h          |     -1.50324     |
|     46 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_h          |     -5.47329     |
|     46 | M3_circ         | test_id_ellip | log_omega | poly2    | log_abs_energy |     -1.84339     |
|     46 | M3_circ         | test_ood_ecc  | log_omega | poly2    | log_abs_energy |     -2.19488     |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_r0         |      0.417847    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_r0         |      0.22705     |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_v0         |      0.236417    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.410344    |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_a          |      0.658061    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_a          |      0.725791    |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | e              |     -0.0249842   |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | e              |   -127.427       |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_h          |      0.523668    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_h          |     -3.0608      |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | ridge    | log_abs_energy |      0.658061    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.725791    |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_r0         |      0.464072    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_r0         |      0.000693319 |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_v0         |      0.23923     |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.423886    |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_a          |      0.774122    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_a          |      0.384303    |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | e              |     -0.0341946   |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | e              |   -121.453       |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_h          |      0.636542    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_h          |     -2.77376     |
|     46 | M1_Ellip_Short  | test_id_ellip | norm      | poly2    | log_abs_energy |      0.774122    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | norm      | poly2    | log_abs_energy |      0.384303    |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_r0         |     -1.69625     |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_r0         |     -1.448       |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_v0         |     -1.54337     |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_v0         |     -0.508069    |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_a          |     -2.04361     |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_a          |     -2.45158     |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | e              |     -0.0304079   |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | e              |   -137.61        |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_h          |     -1.55492     |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_h          |     -9.37379     |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | ridge    | log_abs_energy |     -2.04361     |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.45158     |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_r0         |      0.963194    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_r0         |      0.98118     |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_v0         |      0.97863     |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_v0         |      0.97103     |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_a          |      0.864181    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_a          |      0.763758    |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | e              |      0.136556    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | e              |   -114.343       |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_h          |      0.775634    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_h          |      0.572147    |
|     46 | M1_Ellip_Short  | test_id_ellip | state     | poly2    | log_abs_energy |      0.864181    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.763758    |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_r0         |      0.865309    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.914693    |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_v0         |      0.793228    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.902668    |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_a          |      0.89196     |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_a          |      0.629456    |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | e              |      0.233621    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | e              |    -37.4329      |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_h          |      0.821135    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_h          |     -0.261314    |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.89196     |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.629456    |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_r0         |      0.932301    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_r0         |      0.96055     |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_v0         |      0.897658    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_v0         |      0.944565    |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_a          |      0.892665    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_a          |      0.769943    |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | e              |     -0.109425    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | e              |    -54.7477      |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_h          |      0.923824    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_h          |      0.241398    |
|     46 | M1_Ellip_Short  | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.892665    |
|     46 | M1_Ellip_Short  | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.769943    |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_r0         |      0.449047    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.813058    |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_v0         |      0.210248    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_v0         |     -1.0666      |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_a          |      0.782125    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_a          |      0.240912    |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | e              |     -0.0158514   |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | e              |   -121.998       |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_h          |      0.758068    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_h          |     -1.72322     |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_abs_energy |      0.782125    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_abs_energy |      0.240912    |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_r0         |      0.465403    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_r0         |     -1.15631     |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_v0         |      0.185948    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.72481     |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_a          |      0.810104    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_a          |     -1.24206     |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | e              |      0.0512978   |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | e              |   -100.69        |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_h          |      0.835916    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_h          |     -2.05934     |
|     46 | M1_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_abs_energy |      0.810104    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.24206     |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_r0         |     -1.93967     |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_r0         |     -0.919989    |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_v0         |     -2.12692     |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_v0         |     -0.360887    |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_a          |     -2.12716     |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_a          |     -1.88556     |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | e              |     -0.564275    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | e              |   -154.606       |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_h          |     -2.07232     |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_h          |     -8.98989     |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | ridge    | log_abs_energy |     -2.12716     |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.88556     |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_r0         |      0.998555    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_r0         |      0.997402    |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_v0         |      0.993233    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_v0         |      0.965361    |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_a          |      0.789163    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_a          |      0.852571    |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | e              |    -33.7824      |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | e              |   -337.017       |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_h          |      0.96276     |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_h          |      0.879291    |
|     46 | M1_Ellip_Inv    | test_id_ellip | state     | poly2    | log_abs_energy |      0.789163    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.852571    |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_r0         |      0.145137    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.406881    |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_v0         |     -0.297198    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.0804488   |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_a          |      0.987899    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_a          |      0.969673    |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | e              |      0.506169    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | e              |    -10.6117      |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_h          |      0.938942    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_h          |      0.117958    |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.987899    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.969673    |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_r0         |      0.618507    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_r0         |     -3.09088     |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_v0         |     -0.0522101   |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_v0         |     -3.87874     |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_a          |      0.992753    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_a          |      0.879095    |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | e              |    -17.9493      |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | e              |   -117.12        |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_h          |      0.249954    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_h          |      0.229545    |
|     46 | M1_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.992753    |
|     46 | M1_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.879095    |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_r0         |      0.366217    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_r0         |     -1.17553     |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_v0         |      0.137252    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_v0         |     -1.20109     |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_a          |      0.727954    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_a          |     -0.164012    |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | e              |     -0.00558372  |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | e              |   -118.576       |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_h          |      0.711082    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_h          |     -1.50573     |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | ridge    | log_abs_energy |      0.727954    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -0.164012    |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_r0         |      0.367994    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.970059    |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_v0         |      0.144315    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.630134    |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_a          |      0.700043    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_a          |     -1.02162     |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | e              |      0.0279611   |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | e              |    -98.4206      |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_h          |      0.691894    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_h          |     -1.8826      |
|     46 | M2_Ellip_Inv    | test_id_ellip | norm      | poly2    | log_abs_energy |      0.700043    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.02162     |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_r0         |     -1.74701     |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_r0         |     -0.503059    |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_v0         |     -1.71547     |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_v0         |     -0.12391     |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_a          |     -2.09652     |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_a          |     -1.63787     |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | e              |     -0.753609    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | e              |   -154.297       |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_h          |     -1.99515     |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_h          |     -8.72892     |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | ridge    | log_abs_energy |     -2.09652     |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | ridge    | log_abs_energy |     -1.63787     |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_r0         |      0.993576    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_r0         |      0.994559    |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_v0         |      0.974839    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_v0         |      0.978243    |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_a          |      0.801296    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_a          |      0.503774    |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | e              |    -29.2962      |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | e              |   -517.792       |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_h          |      0.997416    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_h          |      0.840016    |
|     46 | M2_Ellip_Inv    | test_id_ellip | state     | poly2    | log_abs_energy |      0.801296    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | state     | poly2    | log_abs_energy |      0.503774    |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_r0         |      0.826892    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_r0         |      0.556897    |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_v0         |      0.638717    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.237505    |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_a          |      0.995314    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_a          |      0.967605    |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | e              |      0.452204    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | e              |    -70.7447      |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_h          |      0.943852    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_h          |     -1.0586      |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | ridge    | log_abs_energy |      0.995314    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | ridge    | log_abs_energy |      0.967605    |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_r0         |      0.356763    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_r0         |     -1.3228      |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_v0         |     -0.465474    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_v0         |     -1.802       |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_a          |      0.988456    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_a          |      0.855099    |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | e              |      0.236817    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | e              |   -302.502       |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_h          |      0.95074     |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_h          |     -1.0883      |
|     46 | M2_Ellip_Inv    | test_id_ellip | log_omega | poly2    | log_abs_energy |      0.988456    |
|     46 | M2_Ellip_Inv    | test_ood_ecc  | log_omega | poly2    | log_abs_energy |      0.855099    |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_r0         |     -0.627001    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_r0         |     -0.215648    |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_v0         |     -0.314416    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_v0         |     -0.0539349   |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_a          |     -1.33482     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_a          |     -1.33482     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | e              |     -0.00636694  |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | e              |   -118.909       |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_h          |     -1.1407      |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_h          |     -8.14171     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | ridge    | log_abs_energy |     -1.33482     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | ridge    | log_abs_energy |     -1.33482     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_r0         |     -0.627001    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_r0         |     -0.215648    |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_v0         |     -0.314416    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_v0         |     -0.0539349   |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_a          |     -1.33482     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_a          |     -1.33482     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | e              |     -0.00636694  |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | e              |   -118.909       |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_h          |     -1.1407      |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_h          |     -8.14171     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | norm      | poly2    | log_abs_energy |     -1.33482     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | norm      | poly2    | log_abs_energy |     -1.33482     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_r0         |     -2.61916     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_r0         |     -0.455406    |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_v0         |     -1.92007     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_v0         |     -0.0245757   |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_a          |     -4.35548     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_a          |     -2.42607     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | e              |     -0.44317     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | e              |   -125.203       |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_h          |     -3.29208     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_h          |     -8.19829     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | ridge    | log_abs_energy |     -4.35548     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | ridge    | log_abs_energy |     -2.42607     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_r0         |      0.893171    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_r0         |      0.739134    |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_v0         |      0.915727    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_v0         |      0.764833    |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_a          |      0.574737    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_a          |     -0.260341    |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | e              |      0.595494    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | e              |    -34.9145      |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_h          |      0.646032    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_h          |     -3.81194     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | state     | poly2    | log_abs_energy |      0.574737    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | state     | poly2    | log_abs_energy |     -0.260341    |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_r0         |     -0.577493    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_r0         |     -0.296217    |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_v0         |     -0.275213    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_v0         |      0.0484137   |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_a          |     -1.37949     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_a          |     -1.67928     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | e              |     -0.315695    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | e              |    -96.1785      |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_h          |     -0.916063    |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_h          |     -8.24154     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | ridge    | log_abs_energy |     -1.37949     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | ridge    | log_abs_energy |     -1.67928     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_r0         |     -6.56541     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_r0         |     -5.52132     |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_v0         |     -5.77705     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_v0         |     -0.808091    |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_a          |     -9.88949     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_a          |    -16.4947      |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | e              |     -3.83839     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | e              |   -153.957       |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_h          |     -9.89398     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_h          |    -22.8634      |
|     46 | M3_Ellip_SIGReg | test_id_ellip | log_omega | poly2    | log_abs_energy |     -9.88949     |
|     46 | M3_Ellip_SIGReg | test_ood_ecc  | log_omega | poly2    | log_abs_energy |    -16.4947      |

Column selection, standardization and coefficients use only the 200 training orbits. Negative test R2 is retained.

## Frozen controls

|   seed | model      | unchanged   | before_sha256                                                    | after_sha256                                                     |
|--------|------------|-------------|------------------------------------------------------------------|------------------------------------------------------------------|
|     42 | M0_Koopman | True        | 052c6fce9ac365207aa959d015cf8b8ebbee9ecf6d4271cd637a4fb89af04e45 | 052c6fce9ac365207aa959d015cf8b8ebbee9ecf6d4271cd637a4fb89af04e45 |
|     42 | M1_circ    | True        | 580b0442e84abf7ea713f7101402bc244c7029756ab43b88426009e7c3378dd7 | 580b0442e84abf7ea713f7101402bc244c7029756ab43b88426009e7c3378dd7 |
|     42 | M2_circ    | True        | ddaa0f1087bcd0eaa166ff1923577af6d0c90b8e3e53c5e851cda8d680472f28 | ddaa0f1087bcd0eaa166ff1923577af6d0c90b8e3e53c5e851cda8d680472f28 |
|     42 | M3_circ    | True        | 2849b1913c1d733f9316bc26e5241eee6b6c4c4d43c2efdf94cbadf3fb80b030 | 2849b1913c1d733f9316bc26e5241eee6b6c4c4d43c2efdf94cbadf3fb80b030 |
|     43 | M0_Koopman | True        | 65fb35ffe46610f29b191ff96c8a1464b760ee20625f927f5064e2a4d986756f | 65fb35ffe46610f29b191ff96c8a1464b760ee20625f927f5064e2a4d986756f |
|     43 | M1_circ    | True        | a06ac6d73592ba3c4d489248923723dc52f0e65c52e78b3157ac271903806e79 | a06ac6d73592ba3c4d489248923723dc52f0e65c52e78b3157ac271903806e79 |
|     43 | M2_circ    | True        | 9ff8876610acc425105bb6d6835a6f137a48f440c22e65d0a96f1c0440f8b66d | 9ff8876610acc425105bb6d6835a6f137a48f440c22e65d0a96f1c0440f8b66d |
|     43 | M3_circ    | True        | 870c1a16076daa793bc400c14b929d737711a3ac2878eff2f050385d58829e34 | 870c1a16076daa793bc400c14b929d737711a3ac2878eff2f050385d58829e34 |
|     44 | M0_Koopman | True        | daca6c72878ef4052b30842ec9190aefe87142c4f1f8dba06e2ad274e6e4ed99 | daca6c72878ef4052b30842ec9190aefe87142c4f1f8dba06e2ad274e6e4ed99 |
|     44 | M1_circ    | True        | b5a1087a342b678ea2cc819a980b621b84fdc744b3e75463848bc2cc372f4384 | b5a1087a342b678ea2cc819a980b621b84fdc744b3e75463848bc2cc372f4384 |
|     44 | M2_circ    | True        | 283a0407e06a1e94e4a7e1ad6b7588f157483d96da761e230ecdbbdb3077de21 | 283a0407e06a1e94e4a7e1ad6b7588f157483d96da761e230ecdbbdb3077de21 |
|     44 | M3_circ    | True        | 3adce3dd8e440c3b852837e39e210fb67940ac01ae1c9f563d6cfbc60ad5073c | 3adce3dd8e440c3b852837e39e210fb67940ac01ae1c9f563d6cfbc60ad5073c |
|     45 | M0_Koopman | True        | f7edf7a037716ded2fff06fcd6645359a17c7f75e09b27bf7e5672bea76147e5 | f7edf7a037716ded2fff06fcd6645359a17c7f75e09b27bf7e5672bea76147e5 |
|     45 | M1_circ    | True        | de81bae373c1e2078c264adfa79f3cdd1d1152d208fac34e7e42f3900964d770 | de81bae373c1e2078c264adfa79f3cdd1d1152d208fac34e7e42f3900964d770 |
|     45 | M2_circ    | True        | ea779690b6016c44c7e162715d6b5d17ef96fb68704da3fc7fb04c2e844bab93 | ea779690b6016c44c7e162715d6b5d17ef96fb68704da3fc7fb04c2e844bab93 |
|     45 | M3_circ    | True        | 76223421f63491f7141d60a13e3f91bc0820d2cb6191a92a77d0cdf68e49d02d | 76223421f63491f7141d60a13e3f91bc0820d2cb6191a92a77d0cdf68e49d02d |
|     46 | M0_Koopman | True        | e3266997831102c4328e91d9f0e293449b93c6a90e87a2b693fa58c43c3c8f1c | e3266997831102c4328e91d9f0e293449b93c6a90e87a2b693fa58c43c3c8f1c |
|     46 | M1_circ    | True        | 60085075d7b8da7aa2543385050e31cd6a1f26cd6de4f3ddb094e6c0067e6f4b | 60085075d7b8da7aa2543385050e31cd6a1f26cd6de4f3ddb094e6c0067e6f4b |
|     46 | M2_circ    | True        | 8859351e01b135de596748e5e96bcd365cedf30e71ae8fa88d1d4b714cfd64d4 | 8859351e01b135de596748e5e96bcd365cedf30e71ae8fa88d1d4b714cfd64d4 |
|     46 | M3_circ    | True        | 14d2f2c4bc0ac364e268e6e9d60c250c49dc984369e133efa5310a5e7e8149b6 | 14d2f2c4bc0ac364e268e6e9d60c250c49dc984369e133efa5310a5e7e8149b6 |

## Interpretation limits

- A 256-step horizon spans 10.24 time units; the period at a=5 is 70.25. This protocol does not cover a full cycle for every training orbit.

- Paired starts share an orbit identity, which is additional supervision. The short and invariant variants also differ in horizon and number of anchors; this comparison does not isolate the invariance penalty alone.

- Failure on ellipses does not identify a unique norm heuristic. Successful frequency probes do not prove causal use of physical energy.

- log(abs(E)) is an affine transform of log(a). Their probe scores are not independent evidence.

- Circular motion in latent action-angle coordinates can represent nonuniform motion on physical ellipses.

- Finite-window phase regression need not recover the mean orbital frequency even when phase winding per full orbit is correct.

All scalar aggregates below use all five seeds, without filtering failed seeds.

![Four-panel figure](figures/exp3_elliptic_kepler.png)
