# Leak Detection and Localisation with Reduced Pressure Sensors for Water Distribution Networks

Leak **detection** with the **Matrix Profile** and leak **localisation** (pipe + zone of the pipe) with an
**encoder-decoder** and a **hydraulic-model check**, using **only pressure sensors at junctions**.
The number of sensors is halved step by step (all junctions → … → 3) to find out how far it can be reduced before the
accuracy drops - for single leaks and for multiple leaks.

Works for **any EPANET network** (`.inp` file).

Author: **Abhishek Dhumvad**

---

## Requirements

- Linux (packaged `main.exe`) or any OS with Python 3.9+ (source in `src/`)
- [EPyT](https://github.com/OpenWaterAnalytics/EPyT) (EPANET-Python Toolkit), NumPy, pandas, openpyxl, matplotlib, plotly (optional)
- PyTorch + CUDA (optional): encoder-decoder training on an NVIDIA GPU
- Many CPU cores help most: the EPANET simulations run in parallel

`main.exe` installs everything by itself on the first start (internet needed once).
For the source: `pip install -r requirements.txt`

---

## Networks

| No | Network | Junctions | Pipes | File |
|---|---|---|---|---|
| 1 | Modena | 268 | 317 | `networks/Modena.inp` |
| 2 | L-Town | 782 | 905 | `networks/L-TOWN.inp` |
| 3 | L-Town v2 Model | 782 | 905 | `networks/L-TOWN_v2_Model.inp` |
| 4 | L-Town v2 Real | 782 | 905 | `networks/L-TOWN_v2_Real.inp.xz` (original 177 MB file, compressed) |
| 5 | Hanoi | 31 | 34 | `networks/Hanoi.inp` |
| 6 | D-Town | 399 | 443 | `networks/D-Town.inp` |
| - | Net3 | 92 | 117 | `networks/Net3.inp` |

---

## Run the code

```bash
chmod +x dist/main.exe
./dist/main.exe          # menu: every town with its number and report
./dist/main.exe all      # all 6 networks, one after another
./dist/main.exe 5        # one network (5 = Hanoi -> Hanoi.html)
./dist/main.exe 6 1      # several networks in your own order
```

| Command | Network | Report |
|---|---|---|
| `./main.exe 1` | Modena | `Modena.html` |
| `./main.exe 2` | L-Town | `L-Town.html` |
| `./main.exe 3` | L-Town v2 Model | `L-Town_v2_Model.html` |
| `./main.exe 4` | L-Town v2 Real | `L-Town_v2_Real.html` |
| `./main.exe 5` | Hanoi | `Hanoi.html` |
| `./main.exe 6` | D-Town | `D-Town.html` |

Source version: `python3 src/leak_localisation_pressure_sensors.py 5`
Rebuild `main.exe` after changing the code: `python3 tools/build_main_exe.py`
Slurm cluster: `sbatch scripts/run_slurm.sh`

Each run writes an HTML report and a folder `<network>_pressure_sensor_study/` (Excel tables, trained models).

---

## Method

1. **Simulation** - EPANET (via EPyT), 15-min readings with demand and sensor noise; pressure change = measured − expected.
2. **Data split** - leaks on every pipe (random point, size 2-7 % of the average demand, start time, duration);
   80 / 20 split by scenarios into training / test, validation taken from the training positions.
3. **Sensor placement** - backward elimination: remove, one at a time, the sensor that matters least → nested sets.
4. **Detection** - real-time Matrix Profile, thresholds tuned per sensor set on separate days (no false alarms).
5. **Localisation** - encoder-decoder gives the most likely pipes; a hydraulic-model check picks the best pipe and zone.
6. **Evaluation** - training, validation and test all contain 40 % single-leak days, 40 % multi-leak days
   (different pipes, overlapping and non-overlapping) and 20 % leak-free days.

---

## Results - Net3

Accuracy of the detected leaks (pipe = correct pipe; zone = correct pipe **and** correct zone), complete method:

| Pressure sensors | 92 | 46 | 23 | 12 | 6 | 3 |
|---|---|---|---|---|---|---|
| Training - pipe (%) | 83.6 | 76.3 | 71.9 | 55.6 | 25.0 | 5.3 |
| Training - zone (%) | 70.2 | 63.3 | 56.3 | 36.3 | 15.7 | 0.8 |
| Validation - pipe (%) | 85.5 | 79.4 | 68.2 | 63.2 | 31.0 | 10.7 |
| Validation - zone (%) | 71.7 | 61.7 | 52.6 | 47.1 | 13.5 | 2.5 |
| **Test - pipe (%)** | **85.8** | **79.8** | **71.7** | **60.4** | **33.1** | **7.5** |
| **Test - zone (%)** | **69.6** | **59.8** | **51.4** | **38.8** | **15.7** | **2.3** |

Test set: 1160 days (464 single-leak, 464 multi-leak, 232 leak-free), 1621 leaks.

**Other networks:** testing is in progress for Modena, L-Town, L-Town v2 Model, L-Town v2 Real, Hanoi and D-Town -
results awaited.

---

## Repository structure

```
├── dist/main.exe                  one-file Linux program (code + all networks + automatic set-up)
├── src/                           the complete study (readable source)
├── networks/                      EPANET .inp files
├── tools/build_main_exe.py        rebuilds dist/main.exe from src/ + networks/
├── scripts/run_slurm.sh           batch job for a Slurm cluster
├── results/Net3/                  full Net3 report and Excel tables
├── experiments/                   earlier stage of the project (Drumchapel)
└── requirements.txt
```

---

## References

- Yeh, C.-C. M. et al. (2016). *Matrix Profile I: All Pairs Similarity Joins for Time Series.* IEEE ICDM.
- Kyriakou, M. et al. (2023). *EPyT: An EPANET-Python Toolkit for Smart Water Network Simulations.* Journal of Open Source Software.
- Vrachimis, S. G. et al. (2022). *Battle of the Leakage Detection and Isolation Methods.* Journal of Water Resources Planning and Management.
- Rossman, L. A. *EPANET 2.2 User Manual.* U.S. EPA.

## Citation

```bibtex
@misc{dhumvad2026leak,
  author       = {Abhishek Dhumvad},
  title        = {Leak Detection and Localisation with Reduced Pressure Sensors for Water Distribution Networks},
  year         = {2026},
  howpublished = {\url{https://github.com/AbhishekDhumvad/Project-WDN}}
}
```
