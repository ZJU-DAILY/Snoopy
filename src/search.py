from model import Trainer
trainer = Trainer()
trainer.train()
topk = trainer.args.topk
dataset = trainer.args.datasets
model_ver = trainer.args.version
trainer.search("../check/" + dataset + "/" + model_ver, topk)
