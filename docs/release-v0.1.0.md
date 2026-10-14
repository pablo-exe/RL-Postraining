# Adapter package v0.1.0

The selected direct-GRPO adapter was packaged from run `20261013T233906Z-c468e6`, seed 42, selected global step 200. The ZIP contains only the PEFT config, adapter weights, pinned base-model license, loading notes, and weight checksum; optimizer state and tokenizer files are excluded. The adapter is not a standalone model and loads on top of the pinned Qwen checkpoint.

| Artifact | SHA-256 | Size |
|---|---|---:|
| `grpo-correct-seed-42-adapter-v0.1.0.zip` | `509fb52d7a5b252aad6483a1bf02070372270774c54b6e143686b8400c3afd76` | 68,332,062 bytes |
| `adapter_model.safetensors` inside the ZIP | `5321591aacc2c7b31abc1880eb48d365004f14b63fc16de4715514e7e120175e` | 73,911,112 bytes |

The measured model achieved 16/1,024 (1.56%) exact ID greedy solutions, with a paired change of +0.59 percentage points over the base and a 95% bootstrap interval of −0.39 to +1.56 points. It produced 0/512 exact OOD solutions. Seeds 43 and 44 were stopped by the pre-registered validation gate.

The base checkpoint's Apache License 2.0 text is included in the archive. The immutable model revision is `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

**Publication status:** the package is prepared locally at `dist/grpo-correct-seed-42-adapter-v0.1.0.zip`; it has not yet been uploaded as a GitHub Release asset. The `v0.1.0` source tag is pushed. GitHub redirected the upload page to sign-in, so the ZIP and `.sha256` sidecar still need to be attached to the `v0.1.0` release after the owner signs in. No credentials were entered. The local `dist/` directory is ignored by Git.
