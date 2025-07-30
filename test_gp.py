from pyGPGOMEA import GPGOMEARegressor as GPG
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

SEED=42
np.random.seed(SEED)

# Example using Newton's gravitational law (except for the constant)
X = np.random.random(size=(1024,3)) * [1,4,8] # first column is mass1, second column is mass2, third column is distance
def grav(X):
	y = 6.67 * np.multiply(X[:,0],X[:,1])/np.square(X[:,2])
	return y
y = grav(X)

# Split train & test set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=SEED )

print('Running GP-GOMEA...')
ea = GPG( 
	time=-1, generations=100, evaluations=-1, popsize=1000, functions='+_*_-_p/_sqrt_plog_^2_tanh', tournament=5,
		prob='symbreg', multiobj=False, linearscaling=True, erc=True, classweights = False,
		gomea=False, gomfos='',
		subcross=0.5, submut=0.5, reproduction=0.0,
		sblibtype=False, sbrdo=0.0, sbagx=0.0,
		unifdepthvar=True, elitism=1,
		ims=False, syntuniqinit=1000,
		initmaxtreeheight=4, inittype=False,
		maxtreeheight=12, maxsize=40,
		validation=False,
		coeffmut=False,
		gomcoeffmutstrat=False,
		batchsize=False,
		seed=SEED, parallel=1, caching=False, 
		silent=False, logtofile=False
	
	)				
ea.fit(X_train, y_train)

print("\n")

# get the model and change some operators such as protected division and protected log to be sympy-digestible
model = ea.get_model().replace("p/","/").replace("plog","log")
# let's also call vars with their names
import sympy
model = str(sympy.simplify(model))
model = model.replace("x0","mass1").replace("x1","mass2").replace("x2","dist")
# sometimes due to numerical precision, sympy might be unable to do this
if model == "0":
	print("Warning: sympy couldn't make it due to numerical precision")
	# re-set to non-simplified model
	model = ea.get_model().replace("p/","/").replace("plog","log").replace("x0","mass1").replace("x1","mass2").replace("x2","dist")
print('Model found:', model)
print('Evaluations taken:', ea.get_evaluations()) # care: this is not correct if multiple threads were used when fitting
print('Test RMSE:', np.sqrt( mean_squared_error(y_test, ea.predict(X_test)) ))

quit()

# Further info, tests down here

print('A population member:', ea.get_final_population(X_train)[0])
# Grid search
print('Running Grid Search')
hyperparams = [{ 'popsize': [50, 100], 
	'initmaxtreeheight': [3,5], 
	'gomea': [True, False] }]
ea = GPG( generations=10, parallel=4, ims=False )
gs = GridSearchCV( ea, hyperparams )
gs.fit(X, y)

print('Best hyper-parameter settings:', gs.best_params_)
