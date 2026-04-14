import numpy as np
import matplotlib.pyplot as plt

data = np.load("output/optical_flow_npz/ADFES_F01-Joy-Face Forward.npz")
flow = data["flow"]
motion = data["motion_all"]
apex = int(data["apex_global"])

print(f"Flow shape: {flow.shape}")
print(f"Frames: {len(data['frames'])}")
print(f"Apex at frame {apex}")
print(f"Max motion: {motion.max():.2f}")

plt.figure(figsize=(12, 4))
plt.plot(motion, linewidth=2)
plt.axvline(apex, color='r', linestyle='--', label='Apex')
plt.xlabel('Frame')
plt.ylabel('Motion Magnitude')
plt.title('Expression Arc')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('output/motion_curve.png')
print("Saved output/motion_curve.png")
