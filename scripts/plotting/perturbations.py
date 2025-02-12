import sys
from pathlib import Path
from subprocess import run
import h5py
import matplotlib.pyplot as plt

model = "top-zams-model.GSM"
gyre_template = f"""
&constants
/

&model
  model_type = 'EVOL'  ! Obtain stellar structure from an evolutionary model
  file = '../../{model}'    ! File name of the evolutionary model
  file_format = 'GSM' ! File format of the evolutionary model
/

&mode
  l = 0 ! Harmonic degree
/

&osc
  outer_bound = 'VACUUM' ! Assume the density vanishes at the stellar surface
/

&rot
/

&num
  diff_scheme = 'COLLOC_GL4'
/

&scan
  grid_type = 'LINEAR' ! Scan grid uniform in inverse frequency
  freq_min = 10        ! Minimum frequency to scan from
  freq_max = 11        ! Maximum frequency to scan to
  n_freq = 5          ! Number of frequency points in scan
  freq_units = 'CYC_PER_DAY'
/

&grid
  w_osc = 0 ! Oscillatory region weight parameter
  w_exp = 0  ! Exponential region weight parameter
  w_ctr = 0 ! Central region weight parameter
/


&ad_output
  summary_file = 'summary.hdf5'                         ! File name for summary file
  summary_item_list = 'id,l,n_pg,freq,freq_units,E_norm' ! Items to appear in summary file
  freq_units = 'CYC_PER_DAY'                   	      ! Units of freq output items

  detail_template = '%id_detail.hdf5'
  detail_item_list = 'x,y_1,y_2,y_3,y_4,eul_P,lag_P,P,rho,M_r,R_star'
/

&nad_output
/
"""

p = Path("test-data/generated/perturbations/")

if len(sys.argv) == 2 and sys.argv[1] == "rerun":
    input_model = ""

    p.mkdir(exist_ok=True)

    input = f"""
    input test-data/{model}

    scan --frequency-units=cycles-per-day 0 0 10. 11. 5

    post-process

    output {p / "storm.hdf5"} --frequency-units=cycles-per-day --properties frequency,degree,azimuthal-order --profiles radial-coordinate,y1,y2,y3,y4,xi_r,xi_h,pressure,density --model-properties dynamical-frequency
    """

    run(["cargo", "run", "--release", "--bin=storm"], input=input, text=True)

    (p / "gyre-inlist").write_text(gyre_template)

    run(["gyre", "gyre-inlist"], cwd=p)


f = h5py.File(f"test-data/{model}")
f_gyre = h5py.File(p / "1_detail.hdf5")
f_storm = h5py.File(p / "storm.hdf5")

r = f["r"][:]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.plot(r, f_gyre["y_1"]["re"][:])
ax1.twinx().plot(r, f_storm["solutions"]["0"]["y1"][:], c="C1")
ax1.set_title("y1")

ax2.plot(r, f_gyre["y_2"]["re"][:])
ax2.twinx().plot(r, f_storm["solutions"]["0"]["y2"][:], c="C1")
ax2.set_title("y2")

ax3.plot(r, f_gyre["y_3"]["re"][:])
ax3.twinx().plot(r, f_storm["solutions"]["0"]["y3"][:], c="C1")
ax3.set_title("y3")

ax4.plot(r, f_gyre["y_4"]["re"][:])
ax4.twinx().plot(r, f_storm["solutions"]["0"]["y4"][:], c="C1")
ax4.set_title("y4")

fig, ax1 = plt.subplots(1, 1)

ax1.plot(r, f_gyre["eul_P"]["re"][:])
ax1.plot(r, f_gyre["lag_P"]["re"][:])
# ax1.twinx().plot(
#     r,
#     (f_gyre["M_r"] / r**2),
#     c="C1",
# )
G = 6.67430e-8
ax1.twinx().plot(r, (f_storm["solutions"]["0"]["pressure"][:]), "--")
plt.plot(
    r,
    (
        f_storm["solutions"]["0"]["pressure"][:]
        - f_storm["solutions"]["0"]["xi_r"][:] * (G * f["M_r"][:] / r**2) * f["rho"]
    ),
    "--",
)
ax1.set_title("pressure")

plt.show()
