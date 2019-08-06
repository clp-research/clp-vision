# coding: utf-8

from delphin.derivation import UdfTerminal
from nltk.tree import Tree
from collections import defaultdict


def eps_from_ParseResult(this_r):
    this_mrs = this_r.mrs()
    out = []
    for ep in this_mrs.eps():
        if ep.pred.pos in ['n', 'a', 'v']:
            out.append(str(ep.pred))
    return ' '.join(out)


def postproc(instring):
    return instring.replace('_leave_v_1', '_left_a_2')


def dertotree(derivation):
    return Tree(derivation.entity, [dertotree(d) for d in derivation.daughters if type(d) != UdfTerminal])


def parse_and_merge_mrs(mrs, grounding, debug=False):
    # parse the mrs. result is a dict from handle to predicates on it, and a dict of handle equalities (a scoping)
    #  also, while we're at it, we create a structure that links entity variables and groundings
    #  (this is done here so as to not having to store the linkings / string spans from the mrs)
    h2p = defaultdict(list)
    v2g = {}
    for ep in mrs.eps():
        h2p[ep.label].append((ep.intrinsic_variable, ep.pred.string, ep.pred.pos, ep.args))
        if (ep.cfrom, ep.cto) in grounding and ep.intrinsic_variable.startswith('x'):
            v2g[ep.intrinsic_variable] = grounding[(ep.cfrom, ep.cto)]

    # h2p:
    # h1 [('e2', 'unknown', None, {'ARG0': 'e2', 'ARG': 'x4'})]
    # h5 [('x4', '_a_q', 'q', {'ARG0': 'x4', 'RSTR': 'h6', 'BODY': 'h7'})]
    # ...

    # v2g:
    # {'x4': ('boy', ('7358', 'people')),
    # 'x12': ('jet', ('7359', 'scene')),
    # 'x18': ('water', ('7359', 'scene')),
    # 'x24': ('pool', ('7360', 'scene'))}
    if debug:
        print("h2p", h2p, "\n--")
        print("v2g", v2g, "\n--")

    heq = {this_hcon.hi: this_hcon.lo for this_hcon in mrs.hcons()}
    # e.g. {'h0': 'h1', 'h6': 'h8', 'h14': 'h16', 'h20': 'h22', 'h26': 'h28'}
    if debug:
        print("heq", heq, "\n--")

    # extract all predicates, per entity variable
    #  (this also turns verbs into unary predicates of their subjects)
    v2p = defaultdict(list)
    for h, preds in h2p.items():
        for var, pred, pos, args in preds:
            if pos == 'q':
                continue
            if var in v2g:
                v2p[var].append(pred)
                continue
            if args.get('ARG1') in v2g and pos in ['a', 'v']:
                v2p[args.get('ARG1')].append(pred)
    # defaultdict(list,
    #             {'x4': ['_young_a_1', '_boy_n_1', '_sit_v_1'],
    #              'x12': ['_jet_n_1'],
    #              'x18': ['_water_n_1'],
    #              'x24': ['_pool_n_of']})
    if debug:
        print("v2p", v2p, "\n--")

    # and link this to the grounding, for learning
    reg2pred = defaultdict(list)
    for ent_var, preds in v2p.items():
        regid = v2g[ent_var][1][0]
        if regid != 0:  # 0 is the code for "non-visual"
            reg2pred[int(regid)].extend(preds)
    # defaultdict(list,
    #             {'7358': ['_young_a_1', '_boy_n_1', '_sit_v_1'],
    #              '7359': ['_jet_n_1', '_water_n_1'],
    #              '7360': ['_pool_n_of']})
    if debug:
        print("reg2pred", reg2pred, "\n--")

    # this parses the relations (= events) from the mrs (in the reduced form)
    # result is a dict from event variable to predicate and arguments
    e2v = {}
    for h, preds in h2p.items():
        for var, pred, pos, args in preds:
            if pos in ['v', 'p']:
                if 'ARG2' in args:
                    e2v[var] = (pred, args['ARG1'], args['ARG2'])
                elif 'ARG1' in args:
                    e2v[var] = (pred, args['ARG1'])
                # this excludes expletives like "it rains" that don't even have an ARG1
    # {'e10': ('_sit_v_1', 'x4'),
    #  'e11': ('_on_p_state', 'e10', 'x12'),
    #  'e23': ('_in_p_state', 'e10', 'x24')}
    if debug:
        print("e2v", e2v, "\n--")

    # this re-ifies prepositions as event arguments...
    #   X sits on Y becomes sits(x), on(x,y), rather than sit(e,x), on(e,y)
    to_be_deleted = []
    out = {}
    for var, (pred, *args) in e2v.items():
        if len(args) < 2:
            continue
        # print(var)
        if args[0].startswith('e') and args[0] in e2v:
            out[var] = (pred, e2v[args[0]][1], args[1])
            to_be_deleted.append(args[0])
        else:
            out[var] = (pred, *args)
    # print(out)
    for v in to_be_deleted:
        out.pop(v, None)
    e2v = out
    # {'e11': ('_on_p_state', 'x4', 'x12'),
    #  'e23': ('_in_p_state', 'x4', 'x24')}
    if debug:
        print("e2v", e2v, "\n--")

    # this finally swaps this around & inserts groundings, to get format for learning
    rel2reg = []
    for _, (rel, argA, argB) in e2v.items():
        if argA in v2g and argB in v2g:
            rel2reg.append((rel, int(v2g[argA][1][0]), int(v2g[argB][1][0])))
    # [('_near_p_state', '21277', '21280')]

    return reg2pred, rel2reg


def parse_flickr_grounding(gcap):
    # parse the flickr30k entity encoding. result is a dict from positions in raw string to region IDs
    current_ground = None
    start_pos = 0
    end_pos = 0
    ground_dict = {}
    for this_word in gcap.split():
        if this_word.startswith('[/'):
            _, entity_id, ent_type = this_word.split('/', 2)
            # [/EN#6729/vehicles/scene the subway]  has 4 /, but only 3 are meaningful!
            current_ground = (entity_id.split('#')[1], ent_type)
            continue
        end_pos = start_pos + len(this_word)
        if current_ground:
            if this_word.endswith(']'):
                end_pos -= 1
                this_word = this_word[:-1]
                ground_dict[(start_pos, end_pos)] = (this_word, current_ground)
                current_ground = None
                start_pos = end_pos + 1
                continue
            ground_dict[(start_pos, end_pos)] = (this_word, current_ground)
        start_pos = end_pos + 1
    return ground_dict


def parse_visgen_grounding(pphr):
    this_word = []
    master_index = 0
    prev_start = 0
    ground_index = {}
    for i, c in enumerate(pphr + ' '):
        if c != ' ':
            this_word.append(c)
        else:
            this_word = ''.join(this_word)
            if '|' in this_word:
                w_parts = this_word.split('|')
                master_index -= len(this_word) - len(w_parts[0])
                ground_index[(prev_start, master_index +
                              (1 if this_word.endswith('.') else 0))] \
                    = (w_parts[0], (w_parts[1], w_parts[2]))
            this_word = []
            prev_start = master_index + 1
        master_index += 1
    return ground_index


def filter_preproc_rel2den(rel2den, threshold=2):
    '''Filter according to pre-processing status.

    threshold = 1 --> keep only utterances for which no pre-processing was triggered
    threshold = 2 --> keep no triggers + results of pre-proc
    ... if you want only the originals, you need to copy over this line (condition is != 1)...
    '''
    return {k: [(p[0], p[1]) for p in v if p[2] < threshold] for k, v in rel2den.items()}


def filter_rel2den(r2d, filelist):
    return {k: [p for p in v if p[0][1] in filelist] for k, v in r2d.items()}
