import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection

class DataPrep(object):
  
    def __init__(self, raw_df: pd.DataFrame, categorical: list, log:list, mixed:dict, general:list, non_categorical:list, integer:list, type:dict, test_ratio:float):
        
        
        self.categorical_columns = categorical
        self.log_columns = log
        self.mixed_columns = mixed
        self.general_columns = general
        self.non_categorical_columns = non_categorical
        self.integer_columns = integer
        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.column_types["general"] = []
        self.column_types["non_categorical"] = []
        self.lower_bounds = {}
        self.label_encoder_list = []
        
        problem = list(type.keys())[0]
        target_col = list(type.values())[0]
        if problem:
            y_real = raw_df[target_col]
            X_real = raw_df.drop(columns=[target_col])
            
            if problem=="Classification":
                X_train_real, _, y_train_real, _ = model_selection.train_test_split(X_real ,y_real, test_size=test_ratio, stratify=y_real,random_state=42)
            else:
                X_train_real, _, y_train_real, _ = model_selection.train_test_split(X_real ,y_real, test_size=test_ratio,random_state=42) 
            
            X_train_real[target_col]= y_train_real

            self.df = X_train_real

        else:
            self.df = raw_df
        self.df = self.df.replace(r' ', np.nan)
        self.df = self.df.fillna('empty')
       
        all_columns= set(self.df.columns)
        irrelevant_missing_columns = set(self.categorical_columns)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)
        
        for i in relevant_missing_columns:
            if i in self.log_columns:
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x=="empty" else x)
                    self.mixed_columns[i] = [-9999999]
            elif i in list(self.mixed_columns.keys()):
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x=="empty" else x )
                    self.mixed_columns[i].append(-9999999)
            else:
                if "empty" in list(self.df[i].values):   
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x=="empty" else x)
                    self.mixed_columns[i] = [-9999999]
        
        if self.log_columns:
            for log_column in self.log_columns:
                valid_indices = []
                for idx,val in enumerate(self.df[log_column].values):
                    if val!=-9999999:
                        valid_indices.append(idx)
                eps = 1
                lower = np.min(self.df[log_column].iloc[valid_indices].values)
                self.lower_bounds[log_column] = lower
                if lower>0: 
                    self.df[log_column] = self.df[log_column].apply(lambda x: np.log(x) if x!=-9999999 else -9999999)
                elif lower == 0:
                    self.df[log_column] = self.df[log_column].apply(lambda x: np.log(x+eps) if x!=-9999999 else -9999999) 
                else:
                    self.df[log_column] = self.df[log_column].apply(lambda x: np.log(x-lower+eps) if x!=-9999999 else -9999999)

        for column_index, column in enumerate(self.df.columns):            
            if column in self.categorical_columns:        
                label_encoder = preprocessing.LabelEncoder()
                self.df[column] = self.df[column].astype(str)
                label_encoder.fit(self.df[column])
                current_label_encoder = dict()
                current_label_encoder['column'] = column
                current_label_encoder['label_encoder'] = label_encoder
                transformed_column = label_encoder.transform(self.df[column])
                self.df[column] = transformed_column
                self.label_encoder_list.append(current_label_encoder)
                self.column_types["categorical"].append(column_index)

                if column in self.general_columns:
                    self.column_types["general"].append(column_index)
            
                if column in self.non_categorical_columns:
                    self.column_types["non_categorical"].append(column_index)
            
            elif column in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[column]
            
            elif column in self.general_columns:
                self.column_types["general"].append(column_index)
            

        super().__init__()
        
    def inverse_prep(self, data, eps=1):
        
        # 1. Create DataFrame:
        #    - Create a Pandas DataFrame using the input data (Numpy array format) 
        #      and self.df.columns (original data column names).
        df_sample = pd.DataFrame(data, columns=self.df.columns)

        # 2. Inverse Label Encoding (Restore Categorical Data):
        #    - Iterate through self.label_encoder_list (label encoder info saved during training).
        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"] # Get saved label encoder
            col_name = self.label_encoder_list[i]["column"]  # Get corresponding column name
            
            # - Convert the column to integer type.
            #   (***If there are NaNs/infs at this point, an error will occur here***)
            df_sample[col_name] = df_sample[col_name].astype(int) 
            
            # - Restore the numeric categorical data to its original string/object format 
            #   using the label encoder's inverse_transform method.
            df_sample[col_name] = le.inverse_transform(df_sample[col_name])

        # 3. Inverse Log Transformation:
        #    - Process columns in self.log_columns (list of columns where log transformation was applied).
        if self.log_columns:
            for i in df_sample: # Iterate through all columns in the DataFrame
                if i in self.log_columns: # If the current column is a log-transformed column
                    lower_bound = self.lower_bounds[i] # Get the saved minimum value (used during log transformation)
                    
                    # - Restore values using the exponential (exp) function, the inverse of log.
                    # - Apply slight adjustment (eps) based on the lower_bound value.
                    #   (***Note: .apply() returns the result, so it must be assigned like df_sample[i] = ... to be reflected.***)
                    #   (***In the original code, assignment was missing in some parts.***)
                    if lower_bound > 0:
                        df_sample[i].apply(lambda x: np.exp(x)) # <-- Assignment missing!
                    elif lower_bound == 0:
                        df_sample[i] = df_sample[i].apply(lambda x: np.ceil(np.exp(x)-eps) if (np.exp(x)-eps) < 0 else (np.exp(x)-eps))
                    else: 
                        df_sample[i] = df_sample[i].apply(lambda x: np.exp(x)-eps+lower_bound)

        # 4. Integer Column Processing:
        #    - Iterate through self.integer_columns (list of columns designated as integer).
        if self.integer_columns:
            for column in self.integer_columns:
                # Branch to use median for 'age' column (or apply to all integer types)
                if column == 'age': # or could just apply to all integer types
                    # --- Modification start ---
                    df_sample[column] = df_sample[column].replace([np.inf, -np.inf], np.nan)
                    
                    # Calculate median of original data
                    median_val = self.df[column].median() 
                    
                    # Fill NaN values with median
                    df_sample[column] = df_sample[column].fillna(median_val)
                    # --- Modification end ---
                else: # Other integer columns not 'age' (apply existing or different method)
                    df_sample[column] = df_sample[column].replace([np.inf, -np.inf], np.nan)
                    mean_val = self.df[column].mean() 
                    df_sample[column] = df_sample[column].fillna(mean_val)

                # Round and convert to integer (Common)
                df_sample[column] = (np.round(df_sample[column].values))
                df_sample[column] = df_sample[column].astype(int)

        # 5. Convert special values (-9999999, 'empty') to NaN:
        #    - Change specific values used in model or data processing to standard missing values (NaN).
        #    - Since this code runs after astype(int), it does not prevent astype(int) errors.
        df_sample.replace(-9999999, np.nan, inplace=True)
        df_sample.replace('empty', np.nan, inplace=True)

        # 6. Return Final DataFrame:
        return df_sample
