"""Run a short representative training benchmark using the training CLI."""
import argparse,subprocess,sys
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/small.yaml'); p.add_argument('--steps',type=int,default=50); p.add_argument('--max-train-samples',type=int,default=256); a=p.parse_args()
print('Benchmark is measured by training; run on target hardware to obtain throughput and peak memory.')
cmd=[sys.executable,'-m','summarizer.train','--config',a.config,'--run-name','benchmark','--max-train-samples',str(a.max_train_samples),'--max-val-samples','32','--epochs','1','--time-budget-minutes','5']
subprocess.run(cmd,check=True)
print('Read measured tokens/sec from results/benchmark/history.csv. Project epoch time using actual training-set size.')
