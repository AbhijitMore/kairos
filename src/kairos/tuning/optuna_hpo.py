import optuna
import logging
from sklearn.model_selection import cross_val_score
from src.kairos.core.pipeline import create_kairos_pipeline

logger = logging.getLogger(__name__)

class OptunaHPO:
    """
    Bayesian Hyperparameter Optimization using Optuna.
    """
    def __init__(self, config, X, y):
        self.config = config
        self.X = X
        self.y = y

    def objective(self, trial):
        # 1. Suggest parameters
        num_leaves = trial.suggest_int('num_leaves', 20, 150)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        feature_fraction = trial.suggest_float('feature_fraction', 0.5, 1.0)
        
        # Update config for this trial
        trial_config = self.config.copy()
        trial_config['model']['lgbm_params']['num_leaves'] = num_leaves
        trial_config['model']['lgbm_params']['learning_rate'] = learning_rate
        trial_config['model']['lgbm_params']['feature_fraction'] = feature_fraction
        
        # 2. Create pipeline
        pipeline = create_kairos_pipeline(trial_config)
        
        # 3. Evaluate (using AUC as proxy for performance)
        # Note: HybridEnsemble fit currently does internally n-fold. 
        # For HPO, we might want a simpler evaluation or single-fold split to save time.
        # For now, let's assume we do 3-fold CV for speed.
        scores = cross_val_score(pipeline, self.X, self.y, cv=3, scoring='roc_auc')
        
        return scores.mean()

    def run_hpo(self, n_trials=20):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        logger.info(f"Best HPO Score: {study.best_value}")
        logger.info(f"Best HPO Params: {study.best_params}")
        
        return study.best_params
