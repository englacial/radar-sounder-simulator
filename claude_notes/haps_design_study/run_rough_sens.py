"""Surface-roughness sensitivity: run an experiment YAML with the sub-facet
surface roughness correlation length overridden (SURF_ROUGH_CL_M). Use a
fresh out_name -- the chunk rid does not encode the roughness constants."""
import sys, runpy
sys.path.insert(0, "tools"); import run_altitude_comparison as rac
rac.SURF_ROUGH_CL_M = float(sys.argv[1]); rac.SURF_ROUGH_SIGMA_M = float(sys.argv[2])
sys.argv = ["run_basal_clutter.py", "--config", sys.argv[3]]
runpy.run_path("tools/run_basal_clutter.py", run_name="__main__")
