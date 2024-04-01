from model import Trainer
trainer = Trainer()
trainer.train()
topk = trainer.args.topk
dataset = trainer.args.datasets
model_path = trainer.args.model_path
trainer.search("../check/" + dataset + model_path, topk)
