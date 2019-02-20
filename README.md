# transE-implementation
My experiments with transE

## Results
[emb_dim = 50, SGD(lr=0.01)]
- Head prediction: Hits@10: 0.3419952261	| MR: 288.1974065
- Tail prediction: Hits@10: 0.41082764808450845 | MR: 198.27008176601041
- Average prediction: Hits@10: 0.37414298048111594 | MR: 243.20541382404224

[emb_dim = 100, SGD(lr=0.01)]
- Head prediction: Hits@10: 0.33789846117384165	| MR: 286.64447867820081
- Tail prediction: Hits@10: 0.41309610468757935 | MR: 199.93064278580013
- Average prediction: Hits@10: 0.3754972829307105 | MR: 243.28756073200049

## Scope of improvement
- Multiple negative samples
- Better optimizer
