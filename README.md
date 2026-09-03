This is a **work in progress** repository with the beginnings of a radar sounder simulator, created to inform choices in stratospheric ice-penetrating radar sounding instrument design.

The main simulator here is a coherent facet-based model with support for multiple layers (surface, firn, internal horizons, and bed) and sub-facet roughness parameterizations.

This work is heavily inspired by two existing lines of simulator development:
* The [`simc`](https://github.com/lpl-tapir/simc) incoherent simulator (developed by Michael Christoffersen for planetary applications) and its use by Joe MacGregor to [simulate cluttergrams](https://github.com/Snow4Flow/survey_planning/blob/main/clutter_cresis_frames.ipynb) for existing OPR radar data)
* The coherent facet-based Stratton-Chu simulation theory developed by [Gerekos et al., 2018](https://ieeexplore.ieee.org/document/8419071) and its sub-facet roughness extension detailed in [Gerekos et al., 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022RS007594).

The purpose of this simulator in particular is to assess the viability of stratospheric radar sounder mission designs. 

High-altitude radar sounders are limited by both signal-to-noise ratio and signal-to-clutter ratio. Only a coherent simulator can accurately assess the off-nadir clutter environment that a radar sounder is likely to encounter.

(See also https://docs.englacial.org/radar-return-statistics-postprocessing/ for the SNR side of the analysis.)

Like our approach to the SNR side, we draw on as much existing data as possible to inform future designs. Basal reflecitivity is calibrated from [Required Surface SNR](https://github.com/englacial/radar-return-statistics-postprocessing/blob/main/docs/1_rssnr_background.md) extracted from existing radar data. Bed topography in Antarctica uses the geostatistical [DEMOGOGN](https://www.gatorglaciology.com/demogorgn) DEM developed by Mickey MacKie and the Gator Glaciology group. Surface roughness parameterizations are fit from Operation IceBridge ATM laser altimetry data.

This set of tools is open-source, and you are welcome to (re)-use it. Keep in mind this is under active development. If you're looking for something stable, you may be better off looking at one of the existing radar simulators mentioned above.
