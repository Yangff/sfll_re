import z3
from subprocess import check_output
import sys

VERBOSE = False
ExecPath = './DfX_64bit'
NoOracle = True

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

import re
class F:
    def __init__(self, x):
        rFunc = r"(.*) = (.*)\(([^,]*)(,(.*))*\)"
        r = re.match(rFunc,x).groups()
        self.name = r[0].strip()
        self.func = r[1].strip()
        self.src1 = r[2].strip()
        self.x = x
        if not (r[4] is None):
            self.src2 = r[4].strip()
            self.srcs = [self.src1, self.src2]
        else:
            self.srcs = [self.src1]

#INPUT(x)

def ops(x):
    f = F(x)
    return f.name, f

def runAnalysis(cmd):
    # step1. init datas
    myTimer.startTime("Netlist Process")
    cmd = list(cmd)
    mAllInputs = list(map(lambda x: re.match('INPUT\((.*)\)', x).groups()[0], filter(lambda x: x.startswith('INPUT'), cmd)))
    mOrgInput = list(filter(lambda x:x.startswith('N'), mAllInputs))
    mInput = set(mOrgInput)
    mKeyInput = set(filter(lambda x:x.startswith('keyinput'), mAllInputs))
    mOrgOutput = list(map(lambda x: re.match('OUTPUT\((.*)\)', x).groups()[0], filter(lambda x: x.startswith('OUTPUT'), cmd)))
    mOutput = set(mOrgOutput)
    mOps = {k:v for k,v in map(ops, filter(lambda x: not (x.startswith('INPUT') or x.startswith('OUTPUT')), cmd ))}
    mSym = set()
    mxRef = {}
    for i in mAllInputs:
        mSym.add(i)
    for i in mOutput:
        mSym.add(i)
    for i in mOps:
        mSym.add(mOps[i].name)
        mSym.add(mOps[i].src1)
        for j in mOps[i].srcs:
            if not j in mxRef:
                mxRef[j] = set()
            mxRef[j].add(mOps[i].name)
    myTimer.endTime("Netlist Process")
    myTimer.startTime("Finding Related Inputs")
    # step2. assist funcs
    def findUsage(ky, used, vis):
        if ky in vis:
            return used
        vis.add(ky)
        if ky in mxRef:
            using = mxRef[ky]
            for i in using:
                if not (i in used):
                    used.add(i)
                    findUsage(i, used, vis)
        return used
    def findSource(ky, used): 
        if ky in mOps:
            op = mOps[ky]
            for i in op.srcs:
                if not (i in used):
                    used.add(i)
                    findSource(i, used)
        return used
    def findRelatedInput(ky):
        if ky in mxRef:
            mxKy = mxRef[ky]
            for i in mxKy:
                for j in mOps[i].srcs:
                    newused = set()
                    findSource(j, newused)
                    for k in newused:
                        if k in mInput:
                            return k
        return None
    # step3-1. find HD circuit
    keyRelated = {}
    hdRelated = set()
    for i in mKeyInput:
        keyRelated[i] = set()
        findUsage(i, keyRelated[i], hdRelated)
    that_output = ''
    for i in hdRelated:
        if i in mOutput:
            that_output = i
            break

    key2In = {k: findRelatedInput(k) for k in mKeyInput}
    criticalInput = key2In.values()
    if (that_output == ''):
        eprint('Cannot find related key of keyinputs, please check the keyinputs.')
        sys.exit(2)
    eprint('HD circuit detacted. That output node is \'' + that_output + '\', with a circuit size of ' + str(len(hdRelated)) + '.')
    eprint('Find ' + str(len(criticalInput)) + ' HD-related inputs, should be the input bit(s) xor with key.')
    if VERBOSE:
        for i in key2In:
            eprint(i + ' xored with ' + key2In[i])

    mSubCircuit = set()
    findSource(that_output, mSubCircuit)
    eprint(that_output + ' related sub-circuit splitted, with size of ' + str(len(mSubCircuit)))
    myTimer.endTime("Finding Related Inputs")
    """f = open('sub_circuit.bench', 'w') 

    for i in mSubCircuit:
        if (i in mOps and not i in hdRelated):
            f.write(mOps[i].x + '\n')

    f.close()

    f = open('sub_circuit_withhd.bench', 'w') 

    for i in mSubCircuit:
        if (i in mOps):
            f.write(mOps[i].x + '\n')

    f.close()"""

    # step 3-2: find FSC2 output/so-called perturb
    # features: 
    # 1. directly used by HD-circuit
    # 2. related to criticalInput only
    myTimer.startTime("Finding Perturb")
    hdRef = set()
    for i in hdRelated:
        if i in mOps:
            for j in mOps[i].srcs:
                if not (j in hdRelated):
                    hdRef.add(j)

    perturb = ''
    criticalInput = set(criticalInput)
    perturbNodes = set()
    for i in hdRef:
        relatedNodes = set()
        findSource(i, relatedNodes)
        perturbNodes = relatedNodes
        relatedNodes = set(filter(lambda x: x in mInput, relatedNodes))
        if relatedNodes <= criticalInput and criticalInput <= relatedNodes:
            perturb = i
            break
    if perturb == '':
        eprint('Cannot find output of FSC circuit, maybe you can specific one.')
        sys.exit(2)
    eprint('Find output of FSC circuit(so called perturb): \'' + perturb + '\'. ')

    # step 3-3: check the output possibility of perturb
    # P[x = 0]
    def getPossibility(x, cached):
        if x in cached:
            return cached[x]
        if x in mInput:
            return 0.5 
        if x in mKeyInput:
            return 0 # should not related to key input!!
        if not (x in mOps):
            return 0.5 # who are you?
        op = mOps[x]
        res = [getPossibility(i, cached) for i in op.srcs]
        if (op.func == 'NOT'):
            cached[x] = 1 - res[0]
        if (op.func == 'BUF'):
            cached[x] = res[0]
        if (op.func == 'AND'):
            cached[x] = 1 - (1-res[0]) * (1-res[1]) # all one
        if (op.func == 'NAND'):
            cached[x] = (1-res[0]) * (1-res[1]) # not all one
        if (op.func == 'OR'):
            cached[x] = res[0] * res[1] # not all zero
        if (op.func == 'NOR'):
            cached[x] = 1 - res[0] * res[1] # all zero
        return cached[x]
    cached_P = {}
    P = getPossibility(perturb, cached_P)
    if P > 0.5:
        should_get = 1
    else:
        should_get = 0
    eprint('Possibility of [\'' + perturb + '\' = 0] is ' + str(P) +', which means we need to solve \'' + perturb + '\' = ' + str(should_get))
    myTimer.endTime("Finding Perturb")
    myTimer.startTime("SAT")
    s = z3.Solver()
    z3Nodes = {}
    def allocNode(name):
        if not (name in z3Nodes):
            z3Nodes[name] = z3.Bool(name)
        return z3Nodes[name]
    perturbNodes.add(perturb)
    for i in perturbNodes:
        if i in mOps:
            op = mOps[i]
            src = allocNode(op.name)
            res = [allocNode(x) for x in op.srcs]
            if (op.func == 'NOT'):
                s.add(src == z3.Not(res[0]))
            if (op.func == 'BUF'):
                s.add(src == res[0])
            if (op.func == 'AND'):
                s.add(src == z3.And(res[0], res[1]))
            if (op.func == 'NAND'):
                s.add(src == z3.Not(z3.And(res[0], res[1])))
            if (op.func == 'OR'):
                s.add(src == z3.Or(res[0], res[1]))
            if (op.func == 'NOR'):
                s.add(src == z3.Not(z3.Or(res[0], res[1])))
    s.add(allocNode(perturb) == (should_get == 1))
    eprint('Start sat solver...')
    sat_result = s.check()
    eprint('result: ' + str(sat_result))
    sat_inputs = {}
    if (str(sat_result) == 'unsat'):
        eprint('Something goes wrong when solving ' + perturb + ' = 1, cannot get an input')
    else:
        m = s.model()
        sat_inputs = {i:m.evaluate(allocNode(i)) for i in criticalInput}
    eprint('Found ' + str(len(sat_inputs)), 'bits of input. Test with oracle.')
    myTimer.endTime("SAT")
    if VERBOSE:
        eprint(sat_inputs)

    # step 4. retrieve all bits
    
    def evalx(sat_inputs, eval_cache, keys, x):
        if x in mInput:
            if x in sat_inputs:
                return sat_inputs[x]
            else:
                return False # otherbits set to 0
        if x in mKeyInput:
            if x in keys:
                return keys[x]
            else:
                return False
        if x in eval_cache:
            return eval_cache[x]
        if not x in mOps:
            eprint('unknown var ' + x)
            return False
        op = mOps[x]
        rs = [evalx(sat_inputs, eval_cache, keys, i) for i in op.srcs]
        if (op.func == 'BUF'):
            eval_cache[x] = rs[0]
        if (op.func == 'NOT'):
            eval_cache[x] = not rs[0]
        if (op.func == 'AND'):
            eval_cache[x] = rs[0] and rs[1]
        if (op.func == 'NAND'):
            eval_cache[x] = not(rs[0] and rs[1])
        if (op.func == 'OR'):
            eval_cache[x] = rs[0] or rs[1]
        if (op.func == 'NOR'):
            eval_cache[x] = not (rs[0] or rs[1])
        return eval_cache[x]

    def createOracleStr(res):
        ss = [ExecPath]
        for i in mOrgInput:
            if i in res:
                if (res[i]):
                    ss.append("1")
                else:
                    ss.append("0")
            else:
                ss.append("0")
        return ss

    def getOracle(x):
        z = zip(x.split(' '), mOrgOutput)
        return {k:v for v, k in z}

    def runOracle(res):
        cmd = createOracleStr(res)
        if VERBOSE:
            eprint('calling ' + " ".join(cmd))
        return getOracle(str(check_output(cmd)))

    # step 4-1: check the correctness of sat
    eval_cache = {}
    keys = {}
    if not NoOracle:
        model_output = evalx(sat_inputs, eval_cache, keys, that_output)
        oracle_output = runOracle(sat_inputs)[that_output]
        eprint('Model with key = 0 get a result of ' + str(model_output) + ' and Oracle returns ' + str(oracle_output == '1'))
        if (model_output == oracle_output):
            eprint('There maybe something wrong with the sat solver/ analysis or maybe key = 0')
            sys.exit()
        eprint('We get an input that causes the circuit having wrong result. Trying to recover the original key')
    else:
        eprint("No oracle, try to recover key directly.")


    def eval_nochace(res, x):
        eval_cached = {}
        keys = {}
        a = evalx(res, eval_cached, keys, x)
        flip = False
        if eval_cached[perturb] == (1 == should_get):
            flip = True
        return a, flip
    def attacker(res):
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
            if (VERBOSE or (icnt % 10 == 0)) and not TIMING:
                eprint('Working on ' + str(icnt) + '/' + str(allTask))
            res[i] = not res[i]
            r1 = runOracle(res)[that_output] == '1'
            if VERBOSE:
                eprint('Oracle returns ' + str(r1))
            r2, f = eval_nochace(res, that_output)
            if (f == True):
                flips.append(i)
            if VERBOSE:
                eprint('Model returns ' + str(r2))
            if (r1 == r2):
                equset.append(i)
            else:
                nequset.append(i)
            res[i] = not res[i]
        equset.append(change)
        res[change] = not res[change]
        return (equset, nequset, flips)

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
            if VERBOSE or (icnt % 10 == 0):
                eprint('Working on ' + str(icnt) + '/' + str(allTask))
            res[i] = not res[i]
            r2, f = eval_nochace(res, that_output)
            if (f == True):
                flips.append(i)
                nequset.append(i)
            else:
                equset.append(i)
            if VERBOSE:
                eprint('Model returns ' + str(r2))
            res[i] = not res[i]
        equset.append(change)
        res[change] = not res[change]
        return (equset, nequset, flips)

    def getFlippedInput(inp, flip):
        res = {}
        for i in inp:
            if i in flip:
                res[i] = not inp[i]
            else:
                res[i] = inp[i]
        return res
    def rec2str(rec):
        sss = ""
        for i in range(0, len(mKeyInput)):
            if rec["keyinput" + str(i)]:
                sss = sss + "1"
            else:
                sss = sss + "0"
        return sss
    _att = None
    if NoOracle:
        _att = attack_nooracle
    else:
        _att = attacker

    myTimer.startTime("Solve by One PIP")
    equset, nequset, flips = _att(sat_inputs)

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
    myTimer.writeAns("   > H1 = " + str(numHA) + " Key1 = " + idStrA + "\n" + "   > H2 = " + str(numHB) + " Key2 = " + idStrB + '\n')
    myTimer.endTime("Solve by One PIP")


    minH = min(numHA, numHB)

    if (minH > 8):
        # only run gaussian for h > 8
        # len(criticalInput) is required for solve the system
        pips = []
        myTimer.startTime("PIPs for Gaussian")
        z3.set_option('smt.phase_selection',5)
        criticalOrdered = [i for i in criticalInput]
        for i in range(0, len(criticalOrdered)):
            z3.set_option('smt.random_seed',5 + i)
            sat_result = s.check()
            sat_inputs = {}
            m = s.model()
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
        myTimer.writeAns("   > H1 = " + str(numHA) + " Key1 = " + idStrA + "\n" + "   > H2 = " + str(numHB) + " Key2 = " + idStrB + '\n' + "   > KeyG = " + rec2str(rec_keyG) + "\n")

import getopt

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hvti:e:",["ifile=", "exefile="])
    except getopt.GetoptError:
        print('main.py -i <inputfile> -e <oracle path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <inputfile> -e <oracle path>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        if opt == '-v':
            eprint('Verbose log')
            global VERBOSE
            VERBOSE = True
        if opt == '-t':
            global TIMING
            TIMING = True
        if (opt in ("-e", "--exefile")):
            global ExecPath
            global NoOracle
            ExecPath = arg
            NoOracle = False

    orgInputs = ''
    if (inputfile == ''):
        myTimer.newSection("<stdin>")
        orgInputs = sys.stdin.readlines()
    else:
        myTimer.newSection(inputfile)
        orgInputs = open(inputfile, 'r').readlines()

    def run(orgInputs):
        orgInputs = filter(lambda x: not x.startswith('#'), map(lambda x: x.strip(), orgInputs))
        runAnalysis(orgInputs)
    if TIMING:
        for i in range(0, 4):
            run(orgInputs)
        print(myTimer)
    else:
        run(orgInputs)

if __name__ == "__main__":
    main(sys.argv[1:])
