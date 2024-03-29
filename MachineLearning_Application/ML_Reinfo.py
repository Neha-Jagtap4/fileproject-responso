import random 
from typing import List

#Enivornment class : Its is a modle of the world which is external to the agent.It provides observations and rewards to agent

class SampleEnviornment:
	def __init__(self):
		self.steps_left=20

	def get_observations(self)->List[float]:
		return [0.0,0.0,0.0]

	def get_actions(self)->List[int]:
		return [0,1]

	def is_done(self)->bool:
		return self.steps_left==0

	def action(self,action : int)->float:
		if self.is_done():
			raise Exception("Game Over")

		self.steps_left-=1
		return random.random();

#Agent class : A thing or person that tries to get rewards by interactions

class Agent:
	def __init__(self):
		self.total_reward = 0.0

	#Step function environment instance
	#1 : Observe the environment 
	#2 : Make the decision about action based on observations
	#3 : Submit the cation to environment
	#4 : Get the reward for current step

	def step(self,env : SampleEnviornment):
		current_obs = env.get_observations()
		print("Observation {}".format(current_obs))
		action = env.get_actions()
		print(action)
		reward = env.action(random.choice(action))
		self.total_reward += reward
		print("Total reward {}".format(self.total_reward))

def main():
	print("--------------Reinforcement Learning---------------")
	env = SampleEnviornment()
	agent = Agent()
	i=0

	while not env.is_done():
		i+=1
		print("Step no {}".format(i))
		agent.step(env)
	print("Total reward got : %.4f"% agent.total_reward)

if __name__ == "__main__":
	main()