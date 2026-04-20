# References

Bibliography of every primary source cited in the codebase. Entries are
grouped by topic; each citation gives author, year, title, publisher,
and the module/function where the method appears.

Format follows the convention *Author Year* in-text (e.g. `Welzl 1991`),
with the full record here. Where a standard body is the author, the
document identifier is given in place of an author name.

---

## Normative standards

- **ASTRA** (2022). *Fachhandbuch Trassee/Geotechnik (FHB T/G), 24 001-10201.*
  Bundesamt für Strassen ASTRA. — Geometric requirements for Swiss
  road infrastructure retaining walls (crown width, batter, thickness,
  embedment). Rule values live in
  `src/ifc_geo_validator/rules/rulesets/astra_fhb_stuetzmauer.yaml`.
- **SIA 262:2013**. *Betonbau.* Schweizerischer Ingenieur- und Architektenverein. —
  Cross-check ruleset at `rules/rulesets/sia_262_stuetzmauer.yaml`.
- **ISO 16739-1:2024**. *Industry Foundation Classes (IFC) for data sharing
  in the construction and facility management industries — Part 1: Data
  schema.* — Input data format. Parsed by
  `core/ifc_parser.py` via IfcOpenShell.
- **SN EN 1990:2002/A1:2005 + AC:2010**. *Grundlagen der Tragwerksplanung.* —
  Background for tolerance derivations referenced in the thesis.
- **SN EN 1992-1-1:2004 + AC:2010**. *Bemessung und Konstruktion von
  Stahlbeton- und Spannbetontragwerken — Teil 1-1.* — §5.3.1 Tabelle 5.1
  defines the geometric boundary between a column (Stütze) and a wall
  (Wand) as plan aspect ≈ 4:1; used in
  `validation/level2._detect_element_role`.

## Mesh geometry and topology

- **Gauss, C. F.** (1813). *Theoria attractionis corporum sphaeroidicorum
  ellipticorum homogeneorum methodo novo tractata.* Commentationes
  societatis regiae scientiarum Gottingensis recentiores, vol. II. —
  Divergence theorem used in `core/geometry.compute_volume`.
- **Ericson, C.** (2004). *Real-Time Collision Detection.* Morgan Kaufmann. —
  Half-edge watertightness test in `core/mesh_converter._check_watertight`
  and `core/geometry` (§12.3).
- **Botsch, M., Kobbelt, L., Pauly, M., Alliez, P., & Lévy, B.** (2010).
  *Polygon Mesh Processing.* A K Peters/CRC Press. — Mesh quality
  diagnostics and vertex welding (§2, §6) used in `core/mesh_converter`.
- **Möller, T., & Trumbore, B.** (1997). *Fast, minimum storage ray-triangle
  intersection.* Journal of Graphics Tools, 2(1), 21–28. — Backbone of
  the Three.js `Raycaster` used for element picking in
  `viz/mesh_viewer.py`.
- **Appel, A.** (1968). *Some techniques for shading machine renderings of
  solids.* Proc. AFIPS Joint Computer Conference, 37–45. — Ray-casting
  foundation cited by the viewer picking pipeline.

## Computational geometry

- **Welzl, E.** (1991). *Smallest enclosing disks (balls and ellipsoids).*
  New Results and New Trends in Computer Science, LNCS 555, 359–370. —
  Bounding-sphere-based FOV-aware zoom-to-fit in
  `viz/mesh_viewer.py::fitDistance`.
- **Shimrat, M.** (1962). *Algorithm 112: Position of point relative to
  polygon.* Communications of the ACM, 5(8), 434. — Point-in-polygon
  test for clearance profile checking in `validation/clearance.py`.
- **de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M.** (2008).
  *Computational Geometry: Algorithms and Applications* (3rd ed.).
  Springer. — Barycentric coordinate robustness for terrain lookups in
  `core/distance._barycentric_2d`.
- **Farin, G.** (2002). *Curves and Surfaces for CAGD: A Practical Guide*
  (5th ed.). Morgan Kaufmann. — Sagitta approximation
  `δ ≈ L² κ / 8` used for measurement uncertainty on curved walls in
  `validation/level3.py`.
- **do Carmo, M. P.** (1976). *Differential Geometry of Curves and
  Surfaces.* Prentice-Hall. — Curvature conventions for the
  κ = |Δθ|/L profile in `core/face_classifier` and the centerline
  uncertainty analysis in `validation/level3`.
- **Arvo, J., & Kirk, D.** (1989). *A Survey of Ray Tracing Acceleration
  Techniques.* In Glassner (Ed.), *An Introduction to Ray Tracing*,
  Academic Press. — Axis-aligned bounding box Euclidean distance in
  `validation/level5._min_bbox_distance_xy`.

## Classification and pattern recognition

- **Kittler, J., Hatef, M., Duin, R. P. W., & Matas, J.** (1998). *On
  combining classifiers.* IEEE Transactions on Pattern Analysis and
  Machine Intelligence, 20(3), 226–239. — Feature-level fusion with
  fixed weights underpins the five-factor classification confidence
  in `validation/level2._compute_confidence`.

## Statistics and robust estimation

- **Jolliffe, I. T.** (2002). *Principal Component Analysis* (2nd ed.).
  Springer. — Area-weighted PCA for centerline extraction and face
  classification in `core/face_classifier`.
- **Huber, P. J.** (1981). *Robust Statistics.* John Wiley & Sons. —
  MAD noise model (0.6745 scale factor) and trimmed means used across
  `core/anomaly_detection` and `validation/level3`.
- **Tukey, J. W.** (1977). *Exploratory Data Analysis.* Addison-Wesley. —
  Percentile choice (10th/90th) for robust thickness estimators in
  `validation/level3`.
- **Grubbs, F. E.** (1969). *Procedures for detecting outlying observations
  in samples.* Technometrics, 11(1), 1–21. — Outlier test used by
  `core/anomaly_detection`.
- **Hampel, F. R.** (1974). *The influence curve and its role in robust
  estimation.* Journal of the American Statistical Association, 69(346),
  383–393. — MAD-based step detection reference.
- **Rissanen, J.** (1978). *Modeling by shortest data description.*
  Automatica, 14(5), 465–471. — MDL principle invoked in
  `core/face_classifier` for curvature hypothesis testing.
- **Hartigan, J. A., & Wong, M. A.** (1979). *Algorithm AS 136: A K-means
  clustering algorithm.* Journal of the Royal Statistical Society Series
  C, 28(1), 100–108. — Reference for coplanar face clustering.

## Data structures

- **Tarjan, R. E.** (1975). *Efficiency of a good but not linear set union
  algorithm.* Journal of the ACM, 22(2), 215–225. — Union-Find with
  path compression used for connected-component labeling in
  `core/face_classifier`.

## Rendering and viewer

- **Lambert, J. H.** (1760). *Photometria, sive de mensura et gradibus
  luminis, colorum et umbrae.* — Lambertian shading model; the viewer
  uses `MeshLambertMaterial` throughout.
- **Akenine-Möller, T., Haines, E., Hoffman, N., Pesce, A., Iwanicki, M.,
  & Hillaire, S.** (2018). *Real-Time Rendering* (4th ed.). A K Peters/CRC
  Press. — §5.8 "Emission" covers the emissive overlay technique
  used for selection highlight (`#FF69B4`, intensity 0.5) in
  `viz/mesh_viewer.py`.
- **Krijnen, T., & Beetz, J.** (2020). *An IFC schema extension and binary
  serialization format to efficiently integrate point cloud data into
  building models.* Advanced Engineering Informatics, 33, 2017. —
  IfcOpenShell geometry extraction pipeline used by
  `core/mesh_converter`.

## Software

- **IfcOpenShell** (2009–). *Open-source IFC toolkit.* http://ifcopenshell.org
- **OpenCASCADE** (1993–). *3D modeling kernel, OCCT.* https://dev.opencascade.org
- **NumPy**: Harris, C. R. et al. (2020). *Array programming with NumPy.*
  Nature 585, 357–362.
- **Three.js** (2010–). *JavaScript 3D library.* https://threejs.org
- **Streamlit** (2019–). *The fastest way to build data apps.*
  https://streamlit.io
