# coding: utf-8

from delphin.derivation import UdfTerminal
from nltk.tree import Tree


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
