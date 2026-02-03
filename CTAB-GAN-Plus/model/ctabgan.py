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
