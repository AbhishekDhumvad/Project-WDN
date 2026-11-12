# Leak Detection and Localisation with Reduced Pressure Sensors in Water Distribution Networks

Research project by **Abhishek Dhumvad** (M.Tech thesis, ongoing).

The project studies how far the number of **pressure sensors** in a water distribution network (WDN) can be reduced
while leaks can still be **detected** and **localised** - including the **position of the leak along the pipe** -
for single and multiple leaks, on any EPANET network.

> The full implementation and the results are **not public** until the work is published.
> This repository only contains a short, simplified example of the basic ideas (`src/basic_example.py`).

---

## Idea in brief

1. **Leak signatures** - a leak changes the pressures in the network in a characteristic pattern. Hydraulic
   simulations (EPANET) of leaks give these patterns.
2. **Sensor reduction** - sensors are ranked by how much leak information they carry ("one sensor on at a time",
   principal component analysis) and the number of sensors is reduced step by step down to a few.
3. **Detection** - deviations of the measured pressures from normal operation.
4. **Localisation** - an encoder-decoder compresses the pressure pattern of the remaining sensors; the leak pipe and
   the position along it are decided from the compressed code (first the half of the pipe, then the part).

## Basic example

```
pip install -r requirements.txt
cd src
python basic_example.py ../networks/Hanoi.inp
```

It simulates simple leaks on a network, ranks the sensors, trains a small encoder-decoder for a decreasing number of
sensors and prints how often the leak location is identified. It is a teaching example, not the research method.

## Networks

| Network | Junctions | Pipes | File |
|---|---|---|---|
| Hanoi | 31 | 34 | `networks/Hanoi.inp` |
| Net3 | 92 | 117 | `networks/Net3.inp` |
| Modena | 268 | 317 | `networks/Modena.inp` |
| D-Town | 399 | 443 | `networks/D-Town.inp` |
| L-Town | 782 | 905 | `networks/L-TOWN.inp` |
| L-Town v2 Model / Real | 782 | 905 | `networks/L-TOWN_v2_Model.inp`, `networks/L-TOWN_v2_Real.inp.xz` (compressed) |

These are public benchmark networks from the water distribution literature.

## Contact

For questions about the project: Abhishek Dhumvad (GitHub: AbhishekDhumvad).

## License

All rights reserved - see `LICENSE`.
