import numpy as np
import pandas as pd

traveldf = pd.read_csv(r"C:\Users\ADMIN\SEM\DUT\travel survey 기반 가상인구\travel_survey_generated.csv", encoding='utf-8')
kaistdf = pd.read_excel(r"C:\Users\ADMIN\SEM\DUT\kaist_preprocessed.xlsx")

# ======================================
# preprocessing
# ======================================

traveldf = traveldf[traveldf['age'] >= 18] 
conditions = [
    (traveldf['age'] >= 18) & (traveldf['age'] <= 24),
    (traveldf['age'] >= 25) & (traveldf['age'] <= 34),
    (traveldf['age'] >= 35) & (traveldf['age'] <= 44),
    (traveldf['age'] >= 45) & (traveldf['age'] <= 54),
    (traveldf['age'] >= 55) & (traveldf['age'] <= 64),
    (traveldf['age'] >= 65)
]
values=[1,2,3,4,5,6]
traveldf['age'] = np.select(conditions, values)
# Transform PLZ to bundesland codes
plz_to_bundesland = {
    "Sachsen": [(1001, 1936), (2601, 2999), (4001, 4579), (4641, 4889), (8001, 9669)],
    "Brandenburg": [(1941, 1998), (3001, 3253), (4891, 4938), (14401, 14715), (14723, 16949), (17258, 17258), (17261, 17291), (19307, 19357)],
    "Thüringen": [(4581, 4639), (6551, 6578), (7301, 7919), (7920, 7950), (7953, 7980), (96501, 96529), (36401, 36469), (37301, 37359), (98501, 99998)],
    "Sachsen-Anhalt": [(6001, 6548), (6601, 6928), (29401, 29416), (38481, 38489), (38801, 39649)],
    "Berlin": [(10001, 14330)],
    "Mecklenburg-Vorpommern": [(17001, 17256), (17259, 17259), (17301, 17309), (17321, 17321), (17322, 17322), (17328, 17331), (17335, 17335), (17337, 19260), (19273, 19273), (19273, 19306), (19357, 19417), (23921, 23999)],
    "Niedersachsen": [(19271, 19273), (21202, 21449), (21522, 21522), (21601, 21789), (26001, 27478), (27607, 27809), (28784, 29399), (32001, 33829), (34331, 34353), (37194, 37194), (37197, 37199), (37401, 37649), (37689, 37691), (37697, 38479), (38501, 38729), (49001, 49459), (49551, 49849)],
    "Schleswig-Holstein": [(21039, 21039), (21524, 21529), (22145, 22145), (22801, 23919), (24001, 25999), (27483, 27498)],
    "Hamburg": [(20001, 21037), (21039, 21039), (21039, 21170), (22001, 22113), (22115, 22143), (22147, 22786), (27499, 27499)],
    "Bremen": [(27501, 27580), (28001, 28779)],
    "Nordrhein-Westfalen": [(32001, 33829), (34401, 34439), (37651, 37688), (40001, 48432), (48466, 48477), (48481, 48485), (48489, 48496), (48541, 48739), (49461, 49549), (50101, 51597), (51601, 53359), (53581, 53604), (53621, 53949), (57001, 57489), (58001, 59966), (59969, 59969)],
    "Rheinland-Pfalz": [(51598, 51598), (53401, 53579), (53614, 53619), (54181, 55239), (55240, 55252), (55253, 56869), (57501, 57648), (66461, 66509), (66841, 67829), (76711, 76891)],
    "Hessen": [(34001, 34329), (34355, 34355), (34356, 34399), (34441, 36399), (59969, 59969), (60001, 63699), (63701, 63774), (64201, 64753), (64754, 64754), (64754, 65326), (65327, 65391), (65392, 65556), (65583, 65620), (65627, 65627), (65701, 65936), (68501, 68519), (68601, 68649), (69235, 69239), (69430, 69431), (69479, 69488), (69503, 69509)],
    "Saarland": [(66001, 66459), (66511, 66839)],
    "Baden-Württemberg": [(63928, 63928), (63930, 63939), (68001, 68312), (68520, 68549), (68701, 69234), (69240, 69429), (69434, 69434), (69435, 69469), (69510, 69514), (70001, 74592), (74594, 74594), (74594, 76709), (77601, 79879)],
    "Bayern": [(63776, 63776), (63928, 63928), (63930, 63939), (80001, 87490), (87493, 87561), (87571, 87789), (88101, 88146), (88147, 88147), (88147, 88179), (89081, 89081), (89087, 89087), (89201, 89449), (90001, 96489), (97001, 97859), (97888, 97892), (97896, 97896), (97901, 97909)],
}

def get_bundesland_code(plz):
    for bundesland, ranges in plz_to_bundesland.items():
        for start, end in ranges:
            if start <= plz <= end:
                return bundesland
    return np.nan

traveldf['home_province'] = traveldf['home_province'].apply(get_bundesland_code)

traveldf['home_office_days'] = traveldf['home_office_days'].replace({97: 5,
                                                                     0: 5})
traveldf['commute_to_fixed_workplace'] = traveldf['commute_to_fixed_workplace'].replace({3: 2})
traveldf['mode_choice'] = traveldf['mode_choice'].replace({2: 1,
                                                           11: 1,
                                                           13: 1,
                                                           3: 2,
                                                           4: 2,
                                                           5: 2,
                                                           7: 2,
                                                           8: 2,
                                                           9: 2,
                                                           10: 2,
                                                           1: 3,
                                                           12: 3,
                                                           14: 4,
                                                           15: 4,
                                                           16: 4,
                                                           17: 4,
                                                           97:4})
traveldf['occupation'] = traveldf['occupation'].replace({1: 6,
                                                       2: 1,
                                                       3: 2,
                                                       4: 2,
                                                       5: 4,
                                                       6: 3,
                                                       7: 3,
                                                       8: 4,
                                                       9: 5,
                                                       10: 2,
                                                       11: 7,
                                                       97: 7,
                                                       0: 8})

conditions = [
    (kaistdf['home_office_days'].isin([6, 7])),
    (kaistdf['home_office_days'] == 5),
    (kaistdf['home_office_days'].isin([3, 4])),
    (kaistdf['home_office_days'].isin([1, 2])),
    (kaistdf['home_office_days'] == 0)
]
values=[1,2,3,4,5]
kaistdf['home_office_days'] = np.select(conditions, values)
kaistdf['mode_choice'] = kaistdf['mode_choice'].replace({2: 1,
                                                         3: 2,
                                                         4: 3,
                                                         5: 4})
kaistdf['occupation'] = kaistdf['occupation'].replace({9: 8})


# ======================================
# merging
# ======================================

merge_on_cols = [
    'home_province',
    #'sex',
    'age',
    'commute_to_fixed_workplace', 
    'occupation',
    'home_office_days',
    'mode_choice'
]

info_cols_from_kaist = ['preference_SO','prefered_days_SO','preference_DRT']

for col in merge_on_cols:
    traveldf[col] = traveldf[col].astype('Int64')
    kaistdf[col] = kaistdf[col].astype('Int64')
kaistdf[info_cols_from_kaist] = kaistdf[info_cols_from_kaist].astype('Int64')

all_kaist_merge_cols = merge_on_cols + info_cols_from_kaist
kaist_df_unique = kaistdf[all_kaist_merge_cols].drop_duplicates(subset=merge_on_cols) 


merged_df = pd.merge(
    traveldf,
    kaist_df_unique, 
    on=merge_on_cols,
    how='left'
)


null_count = merged_df['preference_SO'].isna().sum()
print(f"'preference_SO' 컬럼의 널값 개수: {null_count}개")

print(merged_df.head())



from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

knn_imputation_columns = [
    'home_province', 'sex', 'age',  
    'commute_to_fixed_workplace', 'occupation', 'home_office_days','mode_choice',
    'preference_SO', 'prefered_days_SO', 'preference_DRT'
]

print("\n결측치 개수 (처리 전):")
print(merged_df[knn_imputation_columns].isnull().sum())


def euclidean_distance(row1, row2, weights=None):

    valid_indices = ~np.isnan(row1) & ~np.isnan(row2)
    
    if not np.any(valid_indices):
        return np.inf

    r1_valid = row1[valid_indices]
    r2_valid = row2[valid_indices]

    if weights is not None:
        weights_valid = weights[valid_indices]
        return np.sqrt(np.sum(weights_valid * ((r1_valid - r2_valid) ** 2)))
    else:
        return np.sqrt(np.sum((r1_valid - r2_valid) ** 2))


def find_k_nearest_neighbors(data_scaled, target_row_index, k, weights=None):

    distances = []
    target_row = data_scaled.loc[target_row_index]

    for i, other_row in data_scaled.iterrows():
        if i == target_row_index:
            continue
        
        weights_array = None
        if weights is not None:
            weights_array = np.array(weights)
        
        dist = euclidean_distance(target_row.values, other_row.values, 
                                  weights=weights_array)
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    k_nearest_indices = [idx for idx, dist in distances[:k]]
    return k_nearest_indices


def impute_stochastic_knn_fast(
    df_original, 
    columns_to_impute, 
    n_neighbors, 
    feature_weights=None
):

    df = df_original.copy()
    
    target_cols = ['preference_SO', 'prefered_days_SO', 'preference_DRT'] 
    
    match_cols = [c for c in columns_to_impute if c not in target_cols]
    
    print(f"매칭 변수(X): {match_cols}")
    print(f"타겟 변수(Y): {target_cols}")

    mask_null = df[target_cols].isnull().any(axis=1)
    
    df_recipient = df[mask_null].copy()    
    df_donor = df[~mask_null].copy()     

    if len(df_recipient) == 0:
        print("결측치가 있는 행이 없습니다. 원본을 반환합니다.")
        return df

    X_donor_raw = df_donor[match_cols].fillna(df[match_cols].median())
    X_recipient_raw = df_recipient[match_cols].fillna(df[match_cols].median())
    
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([X_donor_raw, X_recipient_raw]))
    
    X_donor_scaled = scaler.transform(X_donor_raw)
    X_recipient_scaled = scaler.transform(X_recipient_raw)

    if feature_weights is not None:
        
        weight_dict = dict(zip(columns_to_impute, feature_weights))
        
        w_vector = np.array([weight_dict[col] for col in match_cols])
        w_sqrt = np.sqrt(w_vector)
        
        X_donor_scaled = X_donor_scaled * w_sqrt
        X_recipient_scaled = X_recipient_scaled * w_sqrt

    print(f"KNN 검색 시작... (Donor: {len(X_donor_scaled)}명, Recipient: {len(X_recipient_scaled)}명)")

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
    nbrs.fit(X_donor_scaled)
    distances, indices = nbrs.kneighbors(X_recipient_scaled)
    random_picks = np.random.randint(0, n_neighbors, size=len(df_recipient))
    chosen_neighbor_indices = indices[np.arange(len(df_recipient)), random_picks]
    donor_values = df_donor[target_cols].values
    imputed_values = donor_values[chosen_neighbor_indices]
    df.loc[mask_null, target_cols] = imputed_values

    return df


if __name__ == "__main__":
    
    feature_weights = [
    1.5,  # home_province 
    0.5,  # sex 
    1.0,  # age 
    4.0,  # commute_to_fixed_workplace 
    3.0,  # occupation 
    5.0,  # home_office_days 
    4.0,  # mode_choice 
    1.0,  # preference_SO 
    1.0,  # prefered_days_SO
    1.0   # preference_DRT
    ]

    processed_df = impute_stochastic_knn_fast(
        merged_df,
        columns_to_impute=knn_imputation_columns,
        n_neighbors=8, 
        feature_weights=feature_weights
    )
    
    for col in knn_imputation_columns:
        processed_df[col] = processed_df[col].astype("Int64")


    print("=" * 80)
    print("처리 완료")
    print("=" * 80)
    
    print("\n결측치 개수 (처리 후):")
    print(processed_df[knn_imputation_columns].isnull().sum())
    
    print("\n처리된 데이터프레임의 상위 10개 행:")
    print(processed_df.head(10))

    processed_df.to_csv(r"C:\Users\ADMIN\SEM\DUT\travel_KAIST_merged_imputed_w.occupation.csv", index=False, encoding='utf-8-sig')