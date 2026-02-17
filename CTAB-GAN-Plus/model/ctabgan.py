import pandas as pd
import time
from model.pipeline.data_preparation import DataPrep 
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer 

import warnings

warnings.filterwarnings("ignore")

class CTABGAN():

    def __init__(self,
        raw_csv_path="Real_Datasets/Adult.csv",
        test_ratio=0.20,
        categorical_columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
        log_columns=[],
        mixed_columns={'capital-loss': [0.0], 'capital-gain': [0.0]},
        general_columns=["age"],
        non_categorical_columns=[],
        integer_columns=['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
        problem_type={"Classification": "income"},
        condition_column=None, 

        class_dim=(256, 256, 256, 256),
        random_dim=100,
        num_channels=64,
        classifier_dropout_rate=0.5,
        leaky_relu_slope_classifier=0.2, 
        
        batch_size=500, 
        epochs=300, 
        
        lr_g=2e-4, lr_d=2e-4, lr_c=2e-4,
        beta1_g=0.5, beta2_g=0.9,
        beta1_d=0.5, beta2_d=0.9,
        beta1_c=0.5, beta2_c=0.9,
        eps_common=1e-8,
        l2scale_common=1e-5, 
        
        lambda_gp=10.0, 
        
        gumbel_tau=0.2,
        leaky_relu_slope=0.2, 
        gen_leaky_relu_slope=0.2, 
        disc_max_conv_blocks=3,
        gen_max_conv_blocks=3,
        
        lambda_cond_loss=1.0,
        lambda_info_loss=1.0,
        lambda_aux_classifier_loss=1.0,

        ci_discriminator_steps=5, 

        verbose=False):

        self.__name__ = 'CTABGAN'
            

        self.raw_csv_path = raw_csv_path
        self.raw_df = pd.read_csv(raw_csv_path)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.verbose = verbose 
        self.condition_column = condition_column

        self.synthesizer = CTABGANSynthesizer(
            class_dim=class_dim, 
            random_dim=random_dim, 
            num_channels=num_channels, 
            classifier_dropout_rate=classifier_dropout_rate,
            leaky_relu_slope_classifier=leaky_relu_slope_classifier,
            batch_size=batch_size, 
            epochs=epochs, 
            lr_g=lr_g, lr_d=lr_d, lr_c=lr_c,
            beta1_g=beta1_g, beta2_g=beta2_g,
            beta1_d=beta1_d, beta2_d=beta2_d,
            beta1_c=beta1_c, beta2_c=beta2_c,
            eps_common=eps_common,
            l2scale_common=l2scale_common,
            lambda_gp=lambda_gp,
            gumbel_tau=gumbel_tau,
            leaky_relu_slope=leaky_relu_slope,
            gen_leaky_relu_slope=gen_leaky_relu_slope, 
            disc_max_conv_blocks=disc_max_conv_blocks,
            gen_max_conv_blocks=gen_max_conv_blocks,
            lambda_cond_loss=lambda_cond_loss,
            lambda_info_loss=lambda_info_loss,
            lambda_aux_classifier_loss=lambda_aux_classifier_loss,
            ci_discriminator_steps=ci_discriminator_steps,
            condition_column=self.condition_column, 
            verbose=self.verbose 
        )
                
    def fit(self, trial=None): 
        start_time = time.time()
        if self.verbose:
            print("Data preparation stage started...")
        self.data_prep = DataPrep(
            self.raw_df, 
            self.categorical_columns, 
            self.log_columns, 
            self.mixed_columns, 
            self.general_columns, 
            self.non_categorical_columns, 
            self.integer_columns, 
            self.problem_type, 
            self.test_ratio
        )
        if self.verbose:
            print("Data preparation complete. Starting synthesizer fitting...")
        
        print(f"DEBUG: DataPrep df shape: {self.data_prep.df.shape}")
        print(f"DEBUG: DataPrep columns: {self.data_prep.df.columns.tolist()}")


        self.synthesizer.fit(
            train_data_df=self.data_prep.df, 
            categorical_columns=self.data_prep.column_types.get("categorical", []),
            mixed_columns=self.data_prep.column_types.get("mixed", {}),
            general_columns=self.data_prep.column_types.get("general", []),
            non_categorical_columns=self.data_prep.column_types.get("non_categorical", []),
            type_dict=self.problem_type, 
            trial=trial 
        )

        end_time = time.time()
        duration = end_time - start_time
        trial_num_info = f"Trial {trial.number} " if trial else "" 
        
        if self.verbose:
            print(f"CTABGAN model fitting complete for {trial_num_info}in {duration:.2f} seconds.")
        else: 
            print(f'Finished training for {trial_num_info}in {duration:.2f} seconds.')


    
    def save(self, path):
        """Save the CTABGAN model."""
        import torch
        
        state = {
            # CTABGAN init arguments
            'raw_csv_path': self.raw_csv_path,
            'test_ratio': self.test_ratio,
            'categorical_columns': self.categorical_columns,
            'log_columns': self.log_columns,
            'mixed_columns': self.mixed_columns,
            'general_columns': self.general_columns,
            'non_categorical_columns': self.non_categorical_columns,
            'integer_columns': self.integer_columns,
            'problem_type': self.problem_type,
            'condition_column': self.condition_column,
            
            'class_dim': self.synthesizer.class_dim,
            'random_dim': self.synthesizer.random_dim,
            'num_channels': self.synthesizer.num_channels,
            'classifier_dropout_rate': self.synthesizer.classifier_dropout_rate,
            'leaky_relu_slope_classifier': self.synthesizer.leaky_relu_slope_classifier,
            
            'batch_size': self.synthesizer.batch_size,
            'epochs': self.synthesizer.epochs,
            
            'lr_g': self.synthesizer.lr_g, 'lr_d': self.synthesizer.lr_d, 'lr_c': self.synthesizer.lr_c,
            'beta1_g': self.synthesizer.beta1_g, 'beta2_g': self.synthesizer.beta2_g,
            'beta1_d': self.synthesizer.beta1_d, 'beta2_d': self.synthesizer.beta2_d,
            'beta1_c': self.synthesizer.beta1_c, 'beta2_c': self.synthesizer.beta2_c,
            'eps_common': self.synthesizer.eps_common, 'l2scale_common': self.synthesizer.l2scale_common,
            
            'lambda_gp': self.synthesizer.lambda_gp,
            
            'gumbel_tau': self.synthesizer.gumbel_tau,
            'leaky_relu_slope': self.synthesizer.leaky_relu_slope,
            'gen_leaky_relu_slope': self.synthesizer.gen_leaky_relu_slope,
            'disc_max_conv_blocks': self.synthesizer.disc_max_conv_blocks,
            'gen_max_conv_blocks': self.synthesizer.gen_max_conv_blocks,
            
            'lambda_cond_loss': self.synthesizer.lambda_cond_loss,
            'lambda_info_loss': self.synthesizer.lambda_info_loss,
            'lambda_aux_classifier_loss': self.synthesizer.lambda_aux_classifier_loss,
            
            'ci_discriminator_steps': self.synthesizer.ci_discriminator_steps,
            'verbose': self.verbose,
            
            # Synthesizer State
            'synthesizer_state': self.synthesizer.save_state()
        }
        torch.save(state, path)
        if self.verbose:
            print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load a saved CTABGAN model."""
        import torch
        state = torch.load(path)
        
        # Determine params for __init__
        # We extract keys that match __init__ args
        init_params = {
            'raw_csv_path': state.get('raw_csv_path'),
            'test_ratio': state.get('test_ratio'),
            'categorical_columns': state.get('categorical_columns'),
            'log_columns': state.get('log_columns'),
            'mixed_columns': state.get('mixed_columns'),
            'general_columns': state.get('general_columns'),
            'non_categorical_columns': state.get('non_categorical_columns'),
            'integer_columns': state.get('integer_columns'),
            'problem_type': state.get('problem_type'),
            'condition_column': state.get('condition_column'),
            
            'class_dim': state.get('class_dim'),
            'random_dim': state.get('random_dim'),
            'num_channels': state.get('num_channels'),
            'classifier_dropout_rate': state.get('classifier_dropout_rate'),
            'leaky_relu_slope_classifier': state.get('leaky_relu_slope_classifier'),
            
            'batch_size': state.get('batch_size'),
            'epochs': state.get('epochs'),
            
            'lr_g': state.get('lr_g'), 'lr_d': state.get('lr_d'), 'lr_c': state.get('lr_c'),
            'beta1_g': state.get('beta1_g'), 'beta2_g': state.get('beta2_g'),
            'beta1_d': state.get('beta1_d'), 'beta2_d': state.get('beta2_d'),
            'beta1_c': state.get('beta1_c'), 'beta2_c': state.get('beta2_c'),
            'eps_common': state.get('eps_common'), 'l2scale_common': state.get('l2scale_common'),
            
            'lambda_gp': state.get('lambda_gp'),
            
            'gumbel_tau': state.get('gumbel_tau'),
            'leaky_relu_slope': state.get('leaky_relu_slope'),
            'gen_leaky_relu_slope': state.get('gen_leaky_relu_slope'),
            'disc_max_conv_blocks': state.get('disc_max_conv_blocks'),
            'gen_max_conv_blocks': state.get('gen_max_conv_blocks'),
            
            'lambda_cond_loss': state.get('lambda_cond_loss'),
            'lambda_info_loss': state.get('lambda_info_loss'),
            'lambda_aux_classifier_loss': state.get('lambda_aux_classifier_loss'),
            
            'ci_discriminator_steps': state.get('ci_discriminator_steps'),
            'verbose': state.get('verbose', False)
        }
        
        # Initialize new instance
        model = cls(**init_params)
        
        # Load internal state
        model.synthesizer.load_state(state['synthesizer_state'])
        
        return model


    def save(self, path):
        """Save the CTABGAN model."""
        import torch
        
        state = {
            # CTABGAN init arguments
            'raw_csv_path': self.raw_csv_path,
            'test_ratio': self.test_ratio,
            'categorical_columns': self.categorical_columns,
            'log_columns': self.log_columns,
            'mixed_columns': self.mixed_columns,
            'general_columns': self.general_columns,
            'non_categorical_columns': self.non_categorical_columns,
            'integer_columns': self.integer_columns,
            'problem_type': self.problem_type,
            'condition_column': self.condition_column,
            
            'class_dim': self.synthesizer.class_dim,
            'random_dim': self.synthesizer.random_dim,
            'num_channels': self.synthesizer.num_channels,
            'classifier_dropout_rate': self.synthesizer.classifier_dropout_rate,
            'leaky_relu_slope_classifier': self.synthesizer.leaky_relu_slope_classifier,
            
            'batch_size': self.synthesizer.batch_size,
            'epochs': self.synthesizer.epochs,
            
            'lr_g': self.synthesizer.lr_g, 'lr_d': self.synthesizer.lr_d, 'lr_c': self.synthesizer.lr_c,
            'beta1_g': self.synthesizer.beta1_g, 'beta2_g': self.synthesizer.beta2_g,
            'beta1_d': self.synthesizer.beta1_d, 'beta2_d': self.synthesizer.beta2_d,
            'beta1_c': self.synthesizer.beta1_c, 'beta2_c': self.synthesizer.beta2_c,
            'eps_common': self.synthesizer.eps_common, 'l2scale_common': self.synthesizer.l2scale_common,
            
            'lambda_gp': self.synthesizer.lambda_gp,
            
            'gumbel_tau': self.synthesizer.gumbel_tau,
            'leaky_relu_slope': self.synthesizer.leaky_relu_slope,
            'gen_leaky_relu_slope': self.synthesizer.gen_leaky_relu_slope,
            'disc_max_conv_blocks': self.synthesizer.disc_max_conv_blocks,
            'gen_max_conv_blocks': self.synthesizer.gen_max_conv_blocks,
            
            'lambda_cond_loss': self.synthesizer.lambda_cond_loss,
            'lambda_info_loss': self.synthesizer.lambda_info_loss,
            'lambda_aux_classifier_loss': self.synthesizer.lambda_aux_classifier_loss,
            
            'ci_discriminator_steps': self.synthesizer.ci_discriminator_steps,
            'verbose': self.verbose,
            
            # Synthesizer State
            'synthesizer_state': self.synthesizer.save_state(),
            
            # Data Prep State
            'data_prep': self.data_prep
        }
        torch.save(state, path)
        if self.verbose:
            print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load a saved CTABGAN model."""
        import torch
        state = torch.load(path, weights_only=False)
        
        # Determine params for __init__
        init_params = {
            'raw_csv_path': state.get('raw_csv_path'),
            'test_ratio': state.get('test_ratio'),
            'categorical_columns': state.get('categorical_columns'),
            'log_columns': state.get('log_columns'),
            'mixed_columns': state.get('mixed_columns'),
            'general_columns': state.get('general_columns'),
            'non_categorical_columns': state.get('non_categorical_columns'),
            'integer_columns': state.get('integer_columns'),
            'problem_type': state.get('problem_type'),
            'condition_column': state.get('condition_column'),
            
            'class_dim': state.get('class_dim'),
            'random_dim': state.get('random_dim'),
            'num_channels': state.get('num_channels'),
            'classifier_dropout_rate': state.get('classifier_dropout_rate'),
            'leaky_relu_slope_classifier': state.get('leaky_relu_slope_classifier'),
            
            'batch_size': state.get('batch_size'),
            'epochs': state.get('epochs'),
            
            'lr_g': state.get('lr_g'), 'lr_d': state.get('lr_d'), 'lr_c': state.get('lr_c'),
            'beta1_g': state.get('beta1_g'), 'beta2_g': state.get('beta2_g'),
            'beta1_d': state.get('beta1_d'), 'beta2_d': state.get('beta2_d'),
            'beta1_c': state.get('beta1_c'), 'beta2_c': state.get('beta2_c'),
            'eps_common': state.get('eps_common'), 'l2scale_common': state.get('l2scale_common'),
            
            'lambda_gp': state.get('lambda_gp'),
            
            'gumbel_tau': state.get('gumbel_tau'),
            'leaky_relu_slope': state.get('leaky_relu_slope'),
            'gen_leaky_relu_slope': state.get('gen_leaky_relu_slope'),
            'disc_max_conv_blocks': state.get('disc_max_conv_blocks'),
            'gen_max_conv_blocks': state.get('gen_max_conv_blocks'),
            
            'lambda_cond_loss': state.get('lambda_cond_loss'),
            'lambda_info_loss': state.get('lambda_info_loss'),
            'lambda_aux_classifier_loss': state.get('lambda_aux_classifier_loss'),
            
            'ci_discriminator_steps': state.get('ci_discriminator_steps'),
            'verbose': state.get('verbose', False)
        }
        
        # Initialize new instance
        model = cls(**init_params)
        
        # Load internal state
        model.synthesizer.load_state(state['synthesizer_state'])
        model.data_prep = state['data_prep']
        
        return model

    def generate_samples(self, num_samples=None):
        if self.verbose:
            print(f"Generating {num_samples if num_samples else len(self.raw_df)} samples...")
            
        if num_samples is None:
            num_samples = len(self.raw_df)
            
        sample_raw_synthesized = self.synthesizer.sample(num_samples) 
        sample_df_inversed = self.data_prep.inverse_prep(sample_raw_synthesized)
        
        if self.verbose:
            print(f"Sample generation complete. Generated {len(sample_df_inversed)} samples.")
            
        return sample_df_inversed
