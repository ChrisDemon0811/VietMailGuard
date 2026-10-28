# V2.1 dataset configuration (infrastructure only)

`datasets.example.json` is an **empty inventory**. It does not approve a
dataset, infer any raw-label meaning, or make a row training-eligible.
`schema.json` describes the inventory shape; the empty inventory does not
approve an input or make any loaded row training-eligible. V2.1 source-specific
loaders must be invoked explicitly and keep their outputs local.

`vietmailguard.v21.paths.resolve_v21_paths()` reads
`VIETMAILGUARD_V21_DATA_DIR` as the directory containing local source datasets.
If unset, it points to `data/v21/raw/`. An absolute path may be outside this
repository and may contain spaces or Unicode. A relative path is resolved
against the repository root, not the current working directory. Resolving
paths never creates directories, reads files, or copies data.

For example, set the variable to a directory you already control before a
future V2.1 audit command:

```powershell
$env:VIETMAILGUARD_V21_DATA_DIR = '<absolute path to your local datasets>'
```

```bat
set "VIETMAILGUARD_V21_DATA_DIR=<absolute path to your local datasets>"
```

Do not put real email text, private paths, licenses you have not verified, or
API keys into the tracked example. A future private inventory may use
`datasets.local.json`, which is ignored by Git. Derived V2.1 output locations
remain under `data/v21/`, `models/v21_multilingual/`,
`results/v21_multilingual/`, and `reports/v21/`; these are ignored except for
directory markers until a separate release/privacy decision.
