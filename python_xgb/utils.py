
def create_feature_map(fname, features):
    '''Write feature name for xgboost to map 'fn' -> feature name
        Args:
            fname(string): file name
            features(list): feature list
    '''
    with open(fname, 'w') as f:
        for i, feature in enumerate(features):
            f.write('{0}\t{1}\tq\n'.format(i, feature))


def create_feature_imp(fname, f_imp):
    '''Write feature importance file, and sort desc based on importance
        Args:
            fname(string): file name
            f_imp(dict): {feature_name(string): importance(numeric)}
    '''
    with open(fname, 'w') as f:
        for feature, imp in sorted(f_imp.items(), key=lambda v: v[1], reverse=True):
            f.write('{:20} {:.10f}\n'.format(feature, imp))
