"""Execute notebook setup and Parts 0–1; save actual outputs, stop before Part 2."""
from run_part0 import main

if __name__ == "__main__":
    main(stop_heading="## Part 2", expected_cells=3)
