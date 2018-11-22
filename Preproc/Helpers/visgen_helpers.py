from __future__ import division
from operator import itemgetter
from collections import defaultdict


def lod_to_dict(list_of_dicts, getter):
    out_dict = {}
    for list_item in list_of_dicts:
        out_dict[getter(list_item)] = list_item
    return out_dict


def object_id_to_substring(objsd, synsd, object_id):
    this_syns = objsd[object_id]['synsets']
    if len(this_syns) == 0:
        return False
    else:
        this_syns = this_syns[0]  # TODO: Check how often there is more than one
    if this_syns in synsd:
        synidxd = synsd[this_syns]
        return this_syns, synidxd['entity_idx_start'], synidxd['entity_idx_end']
    return False


def append_if_different(indict, key, new_element):
    if len(indict[key]) == 0 or new_element != indict[key][-1]:
        indict[key].append(new_element)


def serialise_region_descr(this_region):
    '''
    Given a region, collect all relations that are annotated for it,
    and try to annotate the region phrase with the ids of the objects
    that the constituents denote.

    For each region, there will only be one annotated phrase (the region
    description with the resolved object IDs), but there may be more
    than one relation for it.
    '''
    #  index the synset and object dictionaries by key
    synsd = lod_to_dict(this_region['synsets'], itemgetter('synset_name'))
    objsd = lod_to_dict(this_region['objects'], itemgetter('object_id'))
    string_annotations = defaultdict(list)

    #  if there aren't any relationships, we're done
    if len(this_region['relationships']) == 0:
        return (None, None, None)

    #  now collect information from each annotated relationship
    all_rel_ids = []
    all_rels = []
    for this_rel in this_region['relationships']:
        obj_id = this_rel['object_id']
        sub_id = this_rel['subject_id']
        predicate = this_rel['predicate']
        rel_id = this_rel['relationship_id']
        syn = None if not this_rel['synsets'] else this_rel['synsets'][0]
        # this just always takes the first synset, if there is any.
        # so this potentially loses information.

        all_rel_ids.append(rel_id)
        all_rels.append((sub_id, predicate, syn, obj_id))

        #  this looks inside the string index substructure, which
        #  can only be accessed via the wordnet synset, for some
        #  reason.. this returns, if successful, a set of positions
        #  inside the string.
        obj = object_id_to_substring(objsd, synsd, obj_id)
        sub = object_id_to_substring(objsd, synsd, sub_id)

        if obj is not False and sub is not False:
            append_if_different(string_annotations, (obj[1], obj[2]),
                                '%d|%s' % (obj_id, obj[0]))
            append_if_different(string_annotations, (sub[1], sub[2]),
                                '%d|%s' % (sub_id, sub[0]))

    this_phrase = this_region['phrase']
    out_phrase = []
    last_pos = 0
    for key in sorted(string_annotations.iterkeys()):
        val = string_annotations[key]
        start, end = key
        annotation = '|'.join(val)
        out_phrase.append(this_phrase[last_pos:start])
        out_phrase.append(this_phrase[start:end] + '|' + annotation)
        last_pos = end
    if len(out_phrase) != 0 and last_pos < len(this_phrase):
        out_phrase.append(this_phrase[last_pos:])
    return (all_rel_ids, all_rels, ''.join(out_phrase))


def empty_to_none(inlist):
    return inlist if inlist else [None]
