def getAnnDict(ann):
    """parses groundnet supporting annotations
    courtesy of https://github.com/volkancirik/groundnet"""
    w2box = {}
    for box in ann.split('|')[1:]:
        if len(box.split(' ')) >= 2:
            phrase = "_".join(box.split(' ')[1:])
            box = box.split(' ')[0]
            w2box[phrase] = box

    for box in ann.split('|')[0].split(',')[1:]:
        if box.strip() != '':
            w2box[box.strip()] = 'b-1'
    return w2box
