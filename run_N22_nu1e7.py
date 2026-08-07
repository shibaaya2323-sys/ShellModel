
from shell_model_N22_nu1e7 import run_simulation

result = run_simulation(
    dt=0.0001,
    total_time=950.0,
    transient_time=250.0,
    sample_interval=1
)

