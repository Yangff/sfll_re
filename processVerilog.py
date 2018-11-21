import z3
import sys
from Parser import VerilogParser
from functools import *

VERBOSE = False

TIMING = False

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

import time

class Timer:
    def __init__(self):
        self.timeReports = {}
        self.ans = {}
        self.reportNames = []
        self.onGoing = {}
        self.nowTitle = ""
    def newSection(self, title):
        if not TIMING:
            return 
        nowTime = time.time()
        if self.nowTitle:
            for i in self.onGoing:
                self.timeReports[self.nowTitle][self.onGoing[i]]["stop"] = nowTime
        self.timeReports[title] = []
        self.nowTitle = title
        self.onGoing = {}
        self.reportNames.append(title)
    def startTime(self, topic):
        if not TIMING:
            return 
        nowTime = time.time()
        if topic in self.onGoing:
            self.timeReports[self.nowTitle][self.onGoing[topic]]["start"] = nowTime
            self.timeReports[self.nowTitle][self.onGoing[topic]]["cnt"] += 1
        else:
            self.timeReports[self.nowTitle].append({"topic":topic,"start":nowTime, "cnt": 1, "sum": 0})
            self.onGoing[topic] = len(self.timeReports[self.nowTitle]) - 1
    def endTime(self, topic):
        if not TIMING:
            return 
        nowTime = time.time()
        st = self.timeReports[self.nowTitle][self.onGoing[topic]]['start']
        self.timeReports[self.nowTitle][self.onGoing[topic]]["sum"] += nowTime - st
    def writeAns(self, ans):
        if not TIMING:
            return
        self.ans[self.nowTitle] = ans
    def __str__(self):
        if not TIMING:
            return ""
        res =  "Time Report\n"
        for i in self.reportNames:
            res += " > " + i + "\n"
            res += self.ans[i]
            for j in self.timeReports[i]:
                res += "   > Topic: " + j['topic'] + "\n"
                res += "     > Time " + str(j['sum'] / j['cnt']) + "s \n"
        return res
myTimer = Timer()
def runAnalysis(doc):
    # step 1. init data
    ops = doc.ops
    print(len(ops))
    return
    opMap = doc.opMap
    inputs = doc.input
    outputs = doc.output
    keyInput = doc.keyInput
    wire = doc.wire
    wireOut = doc.wireOut
    wireIn = doc.wireIn
    # step 1. assist func
    def traceOut(nodes, masks = False):
        result = set()
        def _traceOutMasked(now):
            result.add(now)
            for x in wireIn[now]:
                for y in opMap[x].outputs:
                    if (not (y in result)) and (y in masks):
                        _traceOutMasked(y)
        def _traceOut(now):
            result.add(now)
            for x in wireIn[now]:
                for y in opMap[x].outputs:
                    if (not (y in result)):
                        _traceOut(y)
        if masks:
            for i in nodes:
                _traceOutMasked(i)
        else:
            for i in nodes:
                _traceOut(i)
        return result
    def traceIn(nodes, masks = False):
        result = set()
        def _traceInMasked(now):
            result.add(now)
            if not now in wireOut:
                return 
            x = wireOut[now]
            for y in opMap[x].inputs:
                if (not (y in result)) and (y in masks):
                    _traceInMasked(y)
        def _traceIn(now):
            result.add(now)
            if not now in wireOut:
                return 
            x = wireOut[now]
            for y in opMap[x].inputs:
                if (not (y in result)):
                    _traceIn(y)
        if masks:
            for i in nodes:
                _traceInMasked(i)
        else:
            for i in nodes:
                _traceIn(i)
        return result

    def findRelatedInput(node, masks, n = 1):
        for i in wireIn[node]:
            mOp = opMap[i]
            pi = traceIn(mOp.inputs, masks)
            pi &= inputs
            if n == 1:
                secondary = set()
                for j in mOp.outputs:
                    secondary |= findRelatedInput(j, masks, 0)
                secondary &= inputs
                return pi, secondary
            return pi

    # step 2. find HD circuit
    myTimer.startTime("Finding Target SubCircuit")
    hdCircuit = traceOut(keyInput)
    # print(hdCircuit)
    thatOutput = hdCircuit & outputs

    if not thatOutput:
        eprint('Cannot find related key of keyinputs, please check the keyinputs.')
        sys.exit(2)
    assert len(thatOutput) == 1

    thatOutput = thatOutput.pop()
    subCircuit = traceIn([thatOutput])

    # hdCircuit = hdCircuit and subCircuit

    eprint('HD circuit detacted. That output node is \'' + thatOutput + '\', with a circuit size of ' + str(len(hdCircuit)) + '.')
    eprint('Sub circuit detacted, with a circuit size of ' + str(len(subCircuit)) + "\n")
    myTimer.endTime("Finding Target SubCircuit")
    myTimer.startTime("Finding Related Inputs")
    key2In = {k: findRelatedInput(k, subCircuit) for k in keyInput}
    determined = set()
    und = []
    for k in key2In:
        if len(key2In[k][0]) == 1:
            key2In[k] = key2In[k][0].pop()
            determined.add(key2In[k])
        else:
            und.append(k)
    if und:
        eprint("Warning, strong related node detacted, should be hamming distance instead of key: " + str(und) + "\n")
    for k in und:
        del key2In[k]
        # unable to elimiate
        # I think they are H, not key
    criticalInput = set(key2In.values())
    if VERBOSE:
        for i in key2In:
            eprint(i + ' xored with ' + key2In[i])
    myTimer.endTime("Finding Related Inputs")
    myTimer.startTime("Finding Perturb")
    hdRef = set()
    for wire in hdCircuit:
        ops = wireIn[wire]
        for i in ops:
            if i in opMap:
                op = opMap[i]
                for j in op.inputs:
                    if not (j in hdCircuit):
                        hdRef.add(j)
    perturbs = []
    for i in hdRef:
        relatedIn = traceIn([i], subCircuit) & criticalInput
        if len(relatedIn) == len(criticalInput):
            perturbs.append(i)

    if not perturbs:
        eprint('Cannot find output of FSC circuit, maybe you can specific one.')
        sys.exit(2)
    eprint('Find output(s) of FSC circuit(so called perturbs): ' + str(perturbs) + '. ')

    states = {}

    def getSP(I):
        if I in inputs:
            return 0.5
        if not (I in states):
            op = wireOut[I]
            result = opMap[op].getPossible(getSP)
            for k in result:
                states[k] = result[k]
        return states[I]
    maxAP = 0
    bestP = 0
    perturb = ''
    backupP = []
    for i in perturbs:
        states = {}
        sp = p = getSP(i)
        if VERBOSE:
            eprint(i + " " + str(p))
        if p < 0.5:
            sp = 1 - p
        if sp > 0.9:
            backupP.append((i, p ))
        if sp > maxAP:
            maxAP = sp
            bestP = p
            perturb = i

    if len(backupP) > 1:
        for pp in backupP:
            xins = traceIn([pp[0]], subCircuit)
            usedGates = set()
            for i in xins:
                if i in wireOut:
                    unit = wireOut[i]
                    usedGates.add(opMap[unit].func)
            if 'ADDH' in usedGates and 'ADDH' in usedGates and 'INV' in usedGates and (not 'AO' in usedGates) and (not 'OA' in usedGates) and (not 'AOI' in usedGates) and (not 'OAI' in usedGates):
                perturb, bestP = pp
                break

    should_get = 1
    if bestP > 0.5:
        should_get = 0

    eprint('Possibility of [\'' + perturb + '\' = 1] is ' + str(bestP) +', which means we need to solve \'' + perturb + '\' = ' + str(should_get))
    myTimer.endTime("Finding Perturb")
    """ output sub circuit
    xins = traceIn([perturb], subCircuit)
    nowOp = [opMap[wireOut[perturb]]]
    f = open('sub_' + perturb + '.csv', 'w')
    traced = set()
    traced.add(wireOut[perturb])
    while nowOp:
        now = nowOp.pop(0)
        #print(now.statement())
        cur = [now.func + "___" + now.name]
        for i in now.inputs:
            if i in wireOut:
                cur.append(opMap[wireOut[i]].func + "___" + wireOut[i])
            else:
                cur.append("INPUT___" + i)
        f.write(";".join(cur) + '\n')
        for i in now.inputs:
            if not i in traced:
                if i in xins:
                    if i in wireOut:
                        if not wireOut[i] in traced:
                            traced.add(wireOut[i])
                            nowOp.append(opMap[wireOut[i]]) 
    f.close()
    """
    """ plan B
    xins = traceIn([perturb], subCircuit)
    usingOp = set()
    for i in xins:
        if i in wireOut:
            unit = wireOut[i]
            usingOp.add(opMap[unit])
    
    pasKey = {}
    for ci in criticalInput:
        visInv = {}
        visADD = {}
        markedInv = set()
        def spTraceOut(node, target, marks, vis):
            if not node in xins:
                return False
            hasADD = False
            vis[node] = False
            if node in wireIn:
                for opi in wireIn[node]:
                    nx = opMap[opi]
                    if nx.func in target:
                        vis[node] = True
                        marks.add(nx)
                        return True
                    # print(nx.func)
                    for o in nx.outputs:
                        if o in vis:
                            if vis[o]:
                                vis[node] = True
                        else:
                            if spTraceOut(o, target, marks, vis):
                                vis[node] = True
            
            return vis[node]
        
        spTraceOut(ci, ['INV'], markedInv, visInv)

        markedADD = set()
        spTraceOut(ci, ['ADDF', 'ADDH'], markedADD, visADD)
        pasKey[ci] = False
        for i in markedInv:
            if (i.outputs[0] in visADD):
                pasKey[ci] = True
    #print(markedADD)
    
    # pOps = list(usingOp)
    nowOp = [opMap[wireOut[perturb]]]
    while nowOp:
        now = nowOp.pop(0)
        if now.func.startswith('ADD'):
            continue
        print(now.statement())
        for i in now.inputs:
            if i in xins:
                if i in wireOut:
                    nowOp.append(opMap[wireOut[i]])

        #print(pasKey[ci])
    """
    myTimer.startTime("SAT")
    sat = z3.Solver()
    z3Nodes = {}

    def getZ3(I):
        if not (I in z3Nodes):
            if not I in inputs:
                op = wireOut[I]
                vals = opMap[op].z3Interface(getZ3)
                for k in vals:
                    if not (k in z3Nodes):
                        z3Nodes[k] = z3.Bool(k)
                    sat.add(z3Nodes[k] == vals[k])
            else:
                z3Nodes[I] = z3.Bool(I)
        return z3Nodes[I]

    def rec2str(rec):
        sss = ""
        for i in range(0, len(keyInput)):
            if ("keyinput" + str(i)) in rec:
                if rec["keyinput" + str(i)]:
                    sss = sss + "1"
                else:
                    sss = sss + "0"
            else:
                sss += "0"
        return sss
    getZ3(perturb)
    sat.add(z3Nodes[perturb] == (should_get == 1))

    eprint('Start sat solver...')
    sat_result = sat.check()
    eprint(' > result: ' + str(sat_result))
    sat_inputs = {}
    if (str(sat_result) == 'unsat'):
        eprint('Something goes wrong when solving ' + perturb + ' = 1, cannot get an input')
    else:
        m = sat.model()
        sat_inputs = {i:bool(m.evaluate(z3Nodes[i])) for i in criticalInput}
       # print('n6069', not bool(m.evaluate(z3Nodes['n6069'])))
       # for i in 'n6072 n6071 n6068 n6067 n6065 n6064'.split(' '):
       #     print(i, bool(m.evaluate(z3Nodes[i])))
    eprint('Found ' + str(len(sat_inputs)), 'bits of input. ')

    myTimer.endTime("SAT")
    myTimer.startTime("Solve by One PIP")
    def eval_nochace(res, x):
        state_vals = {}
        def getVal(I):
            if not (I in state_vals):
                if not (I in inputs or I in keyInput):
                    op = wireOut[I]
                    result = opMap[op].eval(getVal)
                    for k in result:
                        state_vals[k] = result[k]
                else:
                    if I in res:
                        return res[I]
                    else:
                        return False # all other bits set to zero
            return state_vals[I]
        
        a = getVal(x)
        flip = False
        if state_vals[perturb] == (1 == should_get):
            flip = True
           # print('flip')
           # print('n6069', not bool(state_vals['n6069']))
           # for i in 'n6072 n6071 n6068 n6067 n6065 n6064'.split(' '):
           #     print(i, bool(state_vals[i]))
        return a, flip
    def getFlippedInput(inp, flip):
        res = {}
        for i in inp:
            if i in flip:
                res[i] = not inp[i]
            else:
                res[i] = inp[i]
        return res
    def attack_nooracle(res):
        icnt = 0
        flips = []
        keys = list(res.keys())
        change = keys[0]
        others = keys[1:]

        equset = []
        nequset = []
        res[change] = not res[change]
        allTask = len(others)
        for i in others:
            icnt = icnt + 1
            if VERBOSE or (icnt % 10 == 0) and not TIMING:
                eprint('Working on ' + str(icnt) + '/' + str(allTask))
            res[i] = not res[i]
            r2, f = eval_nochace(res, thatOutput)
            if (f == True):
                flips.append(i)
                nequset.append(i)
            else:
                equset.append(i)
            if VERBOSE:
                eprint('Model returns ' + str(r2))
                eprint('Flip returns ' + str(f))
            res[i] = not res[i]
        equset.append(change)
        res[change] = not res[change]
        return (equset, nequset, flips)
    pip = {i:sat_inputs[i] for i in sat_inputs}
    equset, nequset, flips = attack_nooracle(pip)

    # type X
    # pasKey = {i:pasKey[key2In[i]] for i in key2In}
    # print(rec2str(pasKey))
    # TypeA. flip on zero, nequset is all one
    numHA = len(nequset)
    inpA = getFlippedInput(sat_inputs, nequset)
    rec_keysA = {i:inpA[key2In[i]] for i in key2In}
    idStrA = rec2str(rec_keysA)
    eprint("Possible result A. H = " + str(numHA))
    eprint("key = " + idStrA)
    if VERBOSE:
        eprint(nequset)
        eprint(rec_keysA)

    # TypeB, flip on one, equset + flipped is all one 
    numHB = len(equset)
    inpB = getFlippedInput(sat_inputs, equset)
    rec_keysB = {i:inpB[key2In[i]] for i in key2In}
    idStrB = rec2str(rec_keysB)
    eprint("Possible result B. H = " + str(numHB))
    eprint("key = " + idStrB)
    if VERBOSE:
        eprint(equset)
        eprint(rec_keysB)
        eprint("flips: ")
        eprint(flips)
    myTimer.writeAns("   > H1 = " + str(numHA) + " Key1 = " + idStrA + "\n" + "   > H2 = " + str(numHB) + " Key2 = " + idStrB + '\n Gate = ' + str(len(ops)) + '\n')
    myTimer.endTime("Solve by One PIP")

    minH = min(numHA, numHB)

    if (minH >= 4):
        # only run gaussian for h >= 4
        # len(criticalInput) is required for solve the system
        pips = []
        myTimer.startTime("PIPs for Gaussian")
        z3.set_option('smt.phase_selection',5)
        criticalOrdered = [i for i in criticalInput]
        for i in range(0, len(criticalInput)):
            sat_result = sat.check()
            sat_inputs = {}
            z3.set_option('smt.random_seed',5 + i)
            m = sat.model()
            ini = [bool(m.evaluate(z3Nodes[i])) for i in criticalOrdered]
            pips.append(ini)
        myTimer.endTime("PIPs for Gaussian")
        myTimer.startTime("Gaussian elimination")
            # eprint(rec2str(map2key))
        nowH = minH
        factors = [list(map(lambda k: 1 if k == 0 else -1, i)) for i in pips]
        cs = [nowH - sum(i) for i in pips]
        # eprint(factors)
        import numpy as np
        a = np.array(factors)
        b = np.array(cs)
        x = np.linalg.solve(a, b)
        xi = {criticalOrdered[i]: abs(x[i] - 1) < 1e-5 for i in range(0, len(x))}
        rec_keyG = {i:xi[key2In[i]] for i in key2In}
        eprint("key = " + rec2str(rec_keyG))
        #print(x)
        myTimer.endTime("Gaussian elimination")
        myTimer.writeAns("   > H1 = " + str(numHA) + " Key1 = " + idStrA + "\n" + "   > H2 = " + str(numHB) + " Key2 = " + idStrB + '\n' + "   > KeyG = " + rec2str(rec_keyG) +  '\n Gate = ' + str(len(ops)) + '\n')

import getopt

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hvti:",["ifile="])
    except getopt.GetoptError:
        print('processVerilog.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('processVerilog.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        if opt == '-v':
            print('Verbose log')
            global VERBOSE
            VERBOSE = True
        if opt == '-t':
            global TIMING
            TIMING = True

    orgInputs = ''

    if (inputfile == ''):
        myTimer.newSection("<stdin>")
        orgInputs = sys.stdin.readlines()
    else:
        import os, glob
        if not os.path.isdir(inputfile):
            myTimer.newSection(inputfile)
            orgInputs = open(inputfile, 'r').readlines()
        else:
            for i in glob.glob(os.path.join(inputfile, '*.v')):
                eprint("Working on " + i)
                myTimer.newSection(i)
                def run():
                    orgInputs = open(i, 'r').readlines()
                    v = VerilogParser()
                    myTimer.startTime("Verilog Process")
                    v.parseVerilog(orgInputs)
                    myTimer.endTime("Verilog Process")
                    runAnalysis(v)
                if TIMING:
                    for j in range(0, 4):
                        run()
                else:
                    run()
            if (TIMING):
                print(myTimer)
            sys.exit(0)
    def run():
        v = VerilogParser()
        myTimer.startTime("Verilog Process")
        v.parseVerilog(orgInputs)
        myTimer.endTime("Verilog Process")
        runAnalysis(v)
    if (TIMING):
        for i in range(0, 2):
            run()
        print(myTimer)
    else:
        run()

if __name__ == "__main__":
    main(sys.argv[1:])