import sys
import h5py
import matplotlib.pyplot as plt
from subprocess import run
from pathlib import Path


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "rerun":
        run(
            ["cargo", "run", "--release", "--bin", "storm"],
            text=True,
            input=STORM_COMMANDS,
            check=True,
        )
        (OUTPUT / "gyre-inlist").write_text(GYRE_INLIST)
        run(["gyre", "gyre-inlist"], check=True, cwd=OUTPUT)

    with h5py.File(OUTPUT / "storm.hdf5") as f:
        storm_freqs = f["frequency"][:]
        storm_degree = f["degree"][:]

    with h5py.File(OUTPUT / "gyre.hdf5") as f:
        gyre_freqs = f["freq"]["re"][:]
        gyre_degree = f["l"][:]
        gyre_radial_order = f["n_pg"][:]

    def plot(degree):
        print(degree)
        plt.plot(
            gyre_radial_order[gyre_degree == degree],
            gyre_freqs[gyre_degree == degree] / storm_freqs[storm_degree == degree]
            - 1.0,
            lw=1,
            label=rf"$\ell = {degree}$",
        )

    plot(0)
    plot(1)
    plot(2)
    plot(5)
    plot(10)

    plt.xlabel(r"Radial order")
    plt.ylabel("Relative frequency difference")
    plt.legend()

    plt.savefig("test-data/generated/plots/difference-with-gyre_1.pdf")


MODEL = Path("test-data/test-model-tams.GSM")
OUTPUT = Path("test-data/generated/difference-with-gyre/")

STORM_COMMANDS = f"""
input {MODEL.absolute()}
scan 1 0 0.027 5 500 --difference-scheme=colloc6 --inverse
scan 2 0 0.045 5 500 --difference-scheme=colloc6 --inverse
scan 5 0 0.105 5 500 --difference-scheme=colloc6 --inverse
scan 10 0 0.185 8 500 --difference-scheme=colloc6 --inverse
scan 0 0 1 250 1000 --difference-scheme=colloc6
scan 1 0 5 250 300 --difference-scheme=colloc6
scan 2 0 5 250 300 --difference-scheme=colloc6
scan 5 0 5 250 300 --difference-scheme=colloc6
scan 10 0 8 250 300 --difference-scheme=colloc6
post-process
output {OUTPUT / "storm.hdf5"} --properties=frequency,radial-order,degree --frequency-units=cycles-per-day
"""

GYRE_INLIST = f"""
&constants
/

&model
  model_type = 'EVOL'  ! Obtain stellar structure from an evolutionary model
  file = '{MODEL.absolute()}'    ! File name of the evolutionary model
  file_format = 'GSM' ! File format of the evolutionary model
/

&mode
  l = 0 ! Harmonic degree
  tag = 'l0'
/

&mode
  l = 1 ! Harmonic degree
  tag = 'l1'
/

&mode
  l = 2 ! Harmonic degree
  tag = 'l2'
/

&mode
  l = 5 ! Harmonic degree
  tag = 'l5'
/

&mode
  l = 10 ! Harmonic degree
  tag = 'l10'
/

&osc
  outer_bound = 'VACUUM' ! Assume the density vanishes at the stellar surface
  reduce_order = .FALSE.
/

&rot
/

&num
  diff_scheme = 'COLLOC_GL6' ! 4th-order collocation scheme for difference equations
  matrix_type = 'BAND'
/

&scan
  grid_type = 'LINEAR' ! Scan grid uniform in inverse frequency
  freq_min = 5        ! Minimum frequency to scan from
  freq_max = 250        ! Maximum frequency to scan to
  n_freq = 300          ! Number of frequency points in scan
  freq_units = 'NONE'
  tag_list = 'l0,l1,l2,l5'
/
&scan
  grid_type = 'LINEAR' ! Scan grid uniform in inverse frequency
  freq_min = 8        ! Minimum frequency to scan from
  freq_max = 250        ! Maximum frequency to scan to
  n_freq = 300          ! Number of frequency points in scan
  freq_units = 'NONE'
  tag_list = 'l10'
/

&scan
  grid_type = 'LINEAR' ! Scan grid uniform in inverse frequency
  freq_min = 1        ! Minimum frequency to scan from
  freq_max = 5        ! Maximum frequency to scan to
  n_freq = 10          ! Number of frequency points in scan
  freq_units = 'NONE'
  tag_list = 'l0'
/

&scan
  grid_type = 'INVERSE' ! Scan grid uniform in inverse frequency
  freq_min = 0.045        ! Minimum frequency to scan from
  freq_max = 5        ! Maximum frequency to scan to
  n_freq = 800          ! Number of frequency points in scan
  freq_units = 'NONE'
  tag_list = 'l2'
/

&scan
  grid_type = 'INVERSE' ! Scan grid uniform in inverse frequency
  freq_min = 0.027        ! Minimum frequency to scan from
  freq_max = 5        ! Maximum frequency to scan to
  n_freq = 800          ! Number of frequency points in scan
  freq_units = 'NONE'
  tag_list = 'l1'
/
&scan
  grid_type = 'INVERSE' ! Scan grid uniform in inverse frequency
  freq_min = 0.105        ! Minimum frequency to scan from
  freq_max = 5        ! Maximum frequency to scan to
  n_freq = 800          ! Number of frequency points in scan
  freq_units = 'NONE'
  tag_list = 'l5'
/
&scan
  grid_type = 'INVERSE' ! Scan grid uniform in inverse frequency
  freq_min = 0.185        ! Minimum frequency to scan from
  freq_max = 8        ! Maximum frequency to scan to
  n_freq = 800          ! Number of frequency points in scan
  freq_units = 'NONE'
  tag_list = 'l10'
/

&grid
  w_osc = 10 ! Oscillatory region weight parameter
  w_exp = 10 ! Exponential region weight parameter
  w_ctr = 10 ! Central region weight parameter
/


&ad_output
  summary_file = 'gyre.hdf5'                         ! File name for summary file
  summary_item_list = 'l,n_pg,freq' ! Items to appear in summary file
  freq_units = 'CYC_PER_DAY'                   	      ! Units of freq output items
/

&nad_output
/
"""

main()
