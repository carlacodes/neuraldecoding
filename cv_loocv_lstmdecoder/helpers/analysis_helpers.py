def apply_filter(df, filter):
    if filter == 'Target trials':
        return df[df['catchTrial'] != 1]
    if filter == 'Catch trials':
        return df[df['catchTrial'] == 1]
    if filter == 'Level cue':
        return df[df['currAtten'] > 0]
    if filter == 'No Level Cue':
        return df[df['currAtten'] == 0]
    if filter == 'Non Correction Trials':
        return df[df['correctionTrial'] == 0]
    if filter == 'CR trials':
        return df[df['response'] == 0 | df['response'] == 1]
    if filter == 'Correction Trials':
        return df[df['correctionTrial'] == 1]
    if filter == 'Sound Right':
        return df[df['side'] == 1]
    if filter == 'Sound Left':
        return df[df['side'] == 0]
    if filter == 'Hit Trials':
        df = apply_filter(df, 'Target trials')
        return df.loc[df.relReleaseTime > 0 & df.relReleaseTime < 2]
    if filter == 'Noise Trials':
        return df.loc[df.currNoiseAtten <= 0]
    if filter == 'Silence Trials':
        return df.loc[df.currNoiseAtten > 60]
    else:
        return f'Filter "{filter}" not found'