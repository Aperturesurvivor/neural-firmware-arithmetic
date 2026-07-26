# Phase 6 development-run manifest

All runs used seed 11701, Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, and the environment recorded
inside each raw JSON result.

| Run | Raw result | Checkpoint SHA-256 | Code record |
| --- | --- | --- | --- |
| v1 | `pilot_v1.json` | `258556622cb5d940a8f25e00031aade8bce014587358e821c63b62db67412e45` | `532b709` |
| v2 | `pilot_v2.json` | `5c33a62fce82b6cf7bab99f7671029f0c9ceca84f804ed368de6afd8bfc12467` | `052148a` |
| v3 | `pilot_v3.json` | `42cfc6d6ca6927267cebcc42ad229c00c02f9dfadcd0e7c0c71e8734f407654d` | archived by `cfd5c95` |
| v4 | `pilot_v4.json` | `869d71099a71a13dc7deca0bcb0a2ad4ece99280f9d925b4238a28483f8ff829` | archived by `cfd5c95` |
| v5 | `pilot_v5.json` | `9c3c9024fa974e597192565a61fadc46bef50fcd89521ec4abd7e92d1317d8e9` | archived by `cfd5c95` |
| v6 | `pilot_v6.json` | `9c3c9024fa974e597192565a61fadc46bef50fcd89521ec4abd7e92d1317d8e9` | `cfd5c95` |

V3-v5 were sequential development runs made in one working session before
the combined implementation was committed. Their complete raw metrics,
configurations, environment records, retained checkpoint hashes, generated
data code, and chronological architecture changes are archived in `cfd5c95`,
but they do not each have a separate execution commit. This limitation is
recorded rather than retroactively assigning them false source commits.
