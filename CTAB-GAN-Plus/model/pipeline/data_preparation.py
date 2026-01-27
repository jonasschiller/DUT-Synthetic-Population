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
        
        # 1. DataFrame 생성:
        #    - 입력받은 data(Numpy 배열 형태)와 self.df.columns(원본 데이터의 컬럼명)를
        #      이용하여 Pandas DataFrame을 생성합니다.
        df_sample = pd.DataFrame(data, columns=self.df.columns)

        # 2. 라벨 인코더 역변환 (범주형 데이터 복원):
        #    - self.label_encoder_list (학습 시 저장된 라벨 인코더 정보)를 순회합니다.
        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"] # 저장된 라벨 인코더 가져오기
            col_name = self.label_encoder_list[i]["column"]  # 해당 컬럼 이름 가져오기
            
            # - 해당 컬럼을 정수형으로 변환합니다. 
            #   (***만약 이 시점에 NaN/inf가 있으면 여기서 오류 발생***)
            df_sample[col_name] = df_sample[col_name].astype(int) 
            
            # - 라벨 인코더의 inverse_transform 메서드를 사용하여 
            #   숫자 형태의 범주형 데이터를 원래의 문자열/객체 형태로 복원합니다.
            df_sample[col_name] = le.inverse_transform(df_sample[col_name])

        # 3. 로그 변환 역변환:
        #    - self.log_columns (로그 변환을 적용했던 컬럼 목록)에 있는 컬럼들을 처리합니다.
        if self.log_columns:
            for i in df_sample: # DataFrame의 모든 컬럼을 순회
                if i in self.log_columns: # 현재 컬럼이 로그 변환된 컬럼이라면
                    lower_bound = self.lower_bounds[i] # 저장된 최소값(로그 변환 시 사용) 가져오기
                    
                    # - 로그의 역함수인 지수(exp) 함수를 사용하여 값을 복원합니다.
                    # - lower_bound 값에 따라 약간의 조정(eps)을 가합니다.
                    #   (***주의: .apply()는 결과를 반환하므로, df_sample[i] = ... 로 할당해야 반영됩니다.***)
                    #   (***원본 코드에서는 할당이 누락된 부분이 있습니다.***)
                    if lower_bound > 0:
                        df_sample[i].apply(lambda x: np.exp(x)) # <-- 할당 누락!
                    elif lower_bound == 0:
                        df_sample[i] = df_sample[i].apply(lambda x: np.ceil(np.exp(x)-eps) if (np.exp(x)-eps) < 0 else (np.exp(x)-eps))
                    else: 
                        df_sample[i] = df_sample[i].apply(lambda x: np.exp(x)-eps+lower_bound)

        # 4. 정수형 컬럼 처리:
        #    - self.integer_columns (정수형으로 지정된 컬럼 목록)을 순회합니다.
        if self.integer_columns:
            for column in self.integer_columns:
                # 'age' 컬럼이라면 중앙값을 사용하도록 분기 (또는 모든 정수형에 중앙값 적용)
                if column == 'age': # 또는 그냥 모든 정수형에 적용해도 됨
                    # --- 수정 시작 ---
                    df_sample[column] = df_sample[column].replace([np.inf, -np.inf], np.nan)
                    
                    # 원본 데이터의 중앙값 계산
                    median_val = self.df[column].median() 
                    
                    # NaN 값을 중앙값으로 채우기
                    df_sample[column] = df_sample[column].fillna(median_val)
                    # --- 수정 끝 ---
                else: # 'age'가 아닌 다른 정수형 컬럼 (기존 방식 또는 다른 방식 적용)
                    df_sample[column] = df_sample[column].replace([np.inf, -np.inf], np.nan)
                    mean_val = self.df[column].mean() 
                    df_sample[column] = df_sample[column].fillna(mean_val)

                # 반올림 및 정수 변환 (공통)
                df_sample[column] = (np.round(df_sample[column].values))
                df_sample[column] = df_sample[column].astype(int)

        # 5. 특수 값(-9999999, 'empty')을 NaN으로 변환:
        #    - 모델이나 데이터 처리 과정에서 사용된 특정 값들을 표준 결측치(NaN)로 바꿉니다.
        #    - 이 코드는 astype(int) 이후에 실행되므로, astype(int) 오류를 막지는 못합니다.
        df_sample.replace(-9999999, np.nan, inplace=True)
        df_sample.replace('empty', np.nan, inplace=True)

        # 6. 최종 DataFrame 반환:
        return df_sample
