from pathlib import Path;
import matplotlib;
matplotlib.use('Agg');
import matplotlib.pyplot as plt;
import numpy as np;

data=np.genfromtxt('logs/episode_rewards.csv', delimiter=',', skip_header=1);
data=np.atleast_2d(data);
episodes=data[:,0].astype(int);
rewards=data[:,1];
window=min(50, max(1, len(rewards)//20));
kernel=np.ones(window)/window;
moving=np.convolve(rewards, kernel, mode='same');
out=Path('assets/reward_plot.png');
out.parent.mkdir(parents=True, exist_ok=True);
plt.figure(figsize=(10,6));
plt.plot(episodes, rewards, color='#4C78A8', alpha=0.35, linewidth=1, label='Episode reward');
plt.plot(episodes, moving, color='#F58518', linewidth=2.5, label='Moving average ('+str(window)+' eps)');
plt.xlabel('Episode');
plt.ylabel('Reward');
plt.title('Training Reward vs Episode');
plt.grid(True, alpha=0.3); 
plt.legend(); 
plt.tight_layout(); 
plt.savefig(out, dpi=150); 
plt.close(); 
print('Saved '+str(out)+' using '+str(len(rewards))+' rewards. Avg='+format(float(np.mean(rewards)), '.2f')+', Best='+format(float(np.max(rewards)), '.2f')+', Last='+format(float(rewards[-1]), '.2f'))