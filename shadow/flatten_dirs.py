def flatten_dirs_one_level(
        basedir: str,
        seasons: list[str] = 'winter summer fall spring'.split()
):
    import itertools
    import os
    import shutil
    dirs = [
        os.path.join(basedir, season)
        for season in seasons
    ]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    path = os.path.join(basedir, 'other')
    if not os.path.exists(path):
        os.makedirs(path)
    season: dict[str, str] = {
        season: os.path.join(season)
        for season in seasons
    }
    listdirs = [
        os.path.join(basedir, dir)
        for dir in os.listdir(basedir)
    ]
    subdirs = [
        (subdir, zoom)
        for subdir in listdirs
        for zoom in os.listdir(subdir)
    ]
    starts = [
        os.path.join(basedir, *nested)
        for nested in subdirs
    ]
    partitions = [
        subdir.rpartition('-')
        for subdir, zoom in subdirs
    ]
    seasons = [
        season.get(partition[2], 'other')
        for partition in partitions
    ]
    nesteds = [
        subdir[1]
        for subdir in subdirs
    ]
    dests = list(map(
        os.path.join,
        itertools.repeat(basedir),
        seasons,
        nesteds
    ))
    for start, dest in zip(starts, dests):
        shutil.copytree(start, dest)
    # map(shutil.copytree, starts, dests)


if __name__ == '__main__':
    flatten_dirs_one_level('/home/arstneio/Downloads/shadows/test/')
