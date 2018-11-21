import z3
from functools import *

MODE_ACC = False  # enum input for higher acc

def portType(group, func):
    if group == 'Y' or group == 'CO' or (group == 'S' and (func == 'ADDF' or func == 'ADDH')):
        return 'output'
    return 'input'


class Logic:
    """
    rules:
        class1:
            out = func(ins)
        class2:
            X1 X2 | ddd
            not by input N
            out = X2(X1(group1), X1(group2), ...)
            len(group1)len(group2)... = ddd
        class3
            MX MXT MXIT
            out = (not) in[s]
        class4
            out = in
        class5
            CO = in1 in2 + in2 in3 + in1 in3
            out = xor(in)
    """
    LogicPrefix = [
        'XOR', 'OR', 'NAND', 'AND','XNOR', 'NOR',  # class 1
        'AOI', 'AO', 
        'OAI', 'OA',  # class 2
        'MXIT','MXT', 'MX', # class 3 
        'BUFH','BUF', 'INV', # class 4
        'ADDF', 'ADDH' # class 5
    ]

    LogicClass = {
        'XOR': 1, 'OR': 1, 'NAND': 1, 'AND': 1,'XNOR': 1, 'NOR': 1,  
        'AOI': 2, 'AO': 2, 
        'OAI': 2, 'OA': 2,  # class 2
        'MXIT': 3,'MXT': 3, 'MX': 3, # class 3 
        'BUFH': 4,'BUF': 4, 'INV': 4, # class 4
        'ADDH': 5, 'ADDF': 6 # class 5/6
    }

    BasicP = {
        'XOR': lambda I1, I2 : 1 - (I1) * (I2) - (1-(I1)) * (1-(I2)),
        'OR': lambda I1, I2 : 1 - (1 - (I1)) * (1 - (I2)),
        'AND': lambda I1, I2 : (I1) * (I2),
        'INV': lambda I1: 1 - (I1),
        'BUF': lambda I1: (I1),
        'MX': lambda I1, I2, SW: (SW) * (I2) + (1-(SW)) * (I1)# SW == 0 => I1
    }

    BasicA = {
        'XOR': lambda I1, I2 : (I1) ^ (I2),
        'OR': lambda I1, I2 : (I1) or (I2),
        'AND': lambda I1, I2 : (I1) and (I2),
        'INV': lambda I1: not (I1),
        'BUF': lambda I1: (I1),
        'MX': lambda I1, I2, SW: (I2) if SW else (I1)
    }

    BasicZ3 = {
        'XOR': lambda I1, I2 : z3.Xor((I1), (I2)),
        'OR': lambda I1, I2 : z3.Or((I1), (I2)),
        'AND': lambda I1, I2 : z3.And((I1), (I2)),
        'INV': lambda I1: z3.Not((I1)),
        'BUF': lambda I1: (I1),
        'MX': lambda I1, I2, SW: z3.Or(z3.And((SW),  (I2)), z3.And(z3.Not((SW)), (I1)))
    }

    def parseFunc(self,fn):
        for i in Logic.LogicPrefix:
            if fn.startswith(i):
                self.func = i
                self.para = fn[len(i):]
                self.para = re.sub('\D','', self.para)
                self.lclass = Logic.LogicClass[self.func]
                return
        assert False # unknown func type !!
    def statement(self):
        a1 = self.func + self.para + " "
        a1 += self.define['name'] + ' ('
        def processArg(arg):
            sub = []
            for i in arg:
                for j in arg[i]:
                    sub.append("." + j['group'] + str(j['id']) + ('N' if j['inv'] else '') + "(" + j['name'] + ')')
            return ', '.join(sub)
        a1 += processArg(self.define['argsIn']) + ', ' + processArg(self.define['argsOut']) + ' );'
        return a1
    def __init__(self, define):
        self.name = define['name']
        # Define {func:string, name:strig, argsIn: Args, argsOut: Args}
        # Args {A: [Arg], B: [Arg], C:[Arg], ...}
        # Arg {group:string, id:int, name: name (name in net), inv: bool, type:input/output}
        self.define = define
        self.inputs = []
        for i in define['argsIn']:
            for j in define['argsIn'][i]:
                self.inputs.append(j['name'])
        self.outputs = []
        for i in define['argsOut']:
            for j in define['argsOut'][i]:
                self.outputs.append(j['name'])
        self.parseFunc(define['func'])
        #if self.name == 'U5069':
        #    print(self.define['argsIn'])
        #    self.define['argsIn']['D'][0].inv = False
        self.acc_mode_cache = {k:[]  for k in self.outputs}
        if (MODE_ACC):
            for i in range(0, 2 ** len(self.inputs)):
                inx = []
                k = i
                for j in range(0, len(self.inputs)):
                    inx.append((k % 2) == 1)
                    k //= 2
                inmap = {k: v for k,v in zip(self.inputs, inx)}
                rst = self.eval(lambda I: inmap[I])
                for k in rst:
                    if rst[k]:
                        self.acc_mode_cache[k].append(inx)

    def __str__(self):
        outs = ""
        outs += " > Logic: " + self.name + "(" + self.func + ", " + self.para + ")" + "\n"
        outs += "  > Inputs: \n"
        innames = []
        for gsi in self.define['argsIn']:
            for ii in self.define['argsIn'][gsi]:
                nn = "N" if ii['inv'] else ""
                outs += "   >" + ii['group'] + str(ii['id']) + " " + (nn) + ": " + ii['name'] + "\n"
                innames.append(ii['name'])
        outs += "  > Outputs: \n"
        statP = {}
        for gsi in self.define['argsOut']:
            for ii in self.define['argsOut'][gsi]:
                nn = "N" if ii['inv'] else ""
                outs += "   >" + ii['group'] + str(ii['id']) + " " + (nn)  + ": " + ii['name'] + "\n"
                statP[ii['name']] = 0
        outs += "  Input = " + ", ".join(innames) + "\n"

        for i in range(0, 2 ** len(innames)):
            k = i
            outs += "  Input = "
            iis = []
            invals = {}
            for j in range(0, len(innames)):
                invals[innames[j]] = ((k % 2) == 1)
                iis.append(str(k % 2))
                k //= 2
            outs += ", ".join(iis) 
            evals = self.eval(lambda I: invals[I])
            for ii in evals:
                if evals[ii]:
                    statP[ii] = statP[ii] + 1
            outs += " => " + str(evals) + "\n"
        outP = self.getPossible(lambda I: 0.5)
        outPa = [statP[i] / (2 ** len(innames)) for i in statP]
        outs += "  OutputP = " + str(outP) + ", OutputPACC = " + str(outPa)
        return outs
    def getRunFunc(self):
        # class 1, 3, 4 only
        assert self.lclass == 1 or self.lclass == 3 or self.lclass == 4
        runFunc = self.func
        postNot = False
        if runFunc == 'XNOR':
            runFunc = 'XOR'
            postNot = True
        if runFunc == 'NAND':
            runFunc = 'AND'
            postNot = True
        if runFunc == 'NOR':
            runFunc = 'OR'
            postNot = True
        if runFunc == 'MXT':
            runFunc = 'MX'
        if runFunc == 'BUFH':
            runFunc = 'BUF'
        if runFunc == 'MXIT':
            runFunc = 'MX'
            postNot = True

        return runFunc, postNot

    def getRunSteps(self):
        # class == 2
        assert self.lclass == 2
        decode = {'A': 'AND', 'O':'OR'}
        s1 = decode[self.func[0]]
        s2 = decode[self.func[1]]
        postNot = len(self.func) == 3 and self.func[2] == 'I'
        return s1,s2,postNot
    # some input maybe invert input
    def getInputWithINV(self, state, fNOT, arg):
        val = state(arg['name'])
        if (arg['inv']):
            val = fNOT(val)
        return val
    # class2 is grouped in groups
    def applyClass2(self, state, fX1, fX2, fNOT, inv):
        s2_inputs = []
        xlogs = ""
        for gsi in self.define['argsIn']:
            gs = self.define['argsIn'][gsi]
            if len(gs) == 1:
                s2_inputs.append(self.getInputWithINV(state, fNOT, gs[0]))
            else:
                s1_grouped_inputs = [self.getInputWithINV(state, fNOT, ga) for ga in gs]
                xlogs += str(s1_grouped_inputs)
                s1_val = reduce(fX1, s1_grouped_inputs)
                s2_inputs.append(s1_val)
        val = reduce(fX2, s2_inputs)
        if inv:
            val = fNOT(val)
        return val

    def getAllInputs(self,state,fNOT):
        ins = []
        for i in self.define['argsIn']:
            argsi = self.define['argsIn'][i]
            for j in argsi:
                ins.append(self.getInputWithINV(state, fNOT, j))
        return ins

    def applyGen(self, state, fGroup):
        if self.lclass == 1:
            runFunc, postNot = self.getRunFunc()

            in_vals = self.getAllInputs(state, fGroup['INV'])
            val = reduce(fGroup[runFunc], in_vals)
            if postNot:
                val = fGroup['INV'](val)
            return {self.outputs[0]: val}
        if self.lclass == 3:
            runFunc, postNot = self.getRunFunc()
            in_vals = ['A', 'B', 'S']
            in_vals = [self.getInputWithINV(state, fGroup['INV'], (self.define['argsIn'][i][0])) for i in in_vals]
            val = fGroup[runFunc](*in_vals)
            if postNot:
                val = fGroup['INV'](val)
            return {self.outputs[0]: val}
        if self.lclass == 2:
            x1, x2, postNot = self.getRunSteps()
            val = self.applyClass2(state, fGroup[x1], fGroup[x2], fGroup['INV'], postNot)
            return {self.outputs[0]: val}
        if self.lclass == 4:
            runFunc, _ = self.getRunFunc()
            val = fGroup[runFunc](*self.getAllInputs(state, fGroup['INV']))
            return {self.outputs[0]: val}
        if self.lclass == 5: # ADDH
            in_vals = ['A', 'B']
            x1,x2 = [self.getInputWithINV(state, fGroup['INV'], (self.define['argsIn'][i][0])) for i in in_vals]
            s = fGroup['XOR'](x1, x2)
            co = fGroup['AND'](x1, x2)
            rst = {'S': s, 'CO': co}
            rst = {self.define['argsOut'][k][0]['name']: rst[k] for k in self.define['argsOut']}
            return rst
            
        if self.lclass == 6: # ADDF
            in_vals = ['A', 'B', 'CI']
            x1,x2,ci = [self.getInputWithINV(state, fGroup['INV'], (self.define['argsIn'][i][0])) for i in in_vals]
            x1xx2 = fGroup['XOR'](x1, x2)
            s = fGroup['XOR'](x1xx2, ci)
            co = fGroup['OR'](fGroup['AND'](x1xx2, ci), fGroup['AND'](x1,x2))
            rst = {'S': s, 'CO': co}
            rst = {self.define['argsOut'][k][0]['name']: rst[k] for k in self.define['argsOut']}
            return rst
    def getPossible(self, state):
        if (MODE_ACC):
            inP = [state(i) for i in self.inputs]
            ouP = {}
            for o in self.outputs:
                sop = 0
                for j in self.acc_mode_cache[o]:
                    op = 1
                    for val, p in zip(self.inputs, inP):
                        if val:
                            op *= p
                        else:
                            op *= 1 - p
                    sop += op
                ouP[o] = sop
            return ouP
        if (self.lclass == 6):
            # special workaround for adder
            in_vals = ['A', 'B', 'CI']
            x1,x2,ci = [state(self.define['argsIn'][i][0]['name']) for i in in_vals]
            x1xx2 = Logic.BasicP['XOR'](x1, x2)
            s = Logic.BasicP['XOR'](x1xx2, ci)
            co = 0
            for v1 in range(0, 2):
                for v2 in range(0, 2):
                    for v3 in range(0, 2):
                        if v1 + v2 + v3 > 1:
                            p1 = x1 if v1 == 1 else 1-x1
                            p2 = x2 if v2 == 1 else 1-x2
                            p3 = ci if v3 == 1 else 1-ci
                            co += p1 * p2 * p3
            rst = {'S': s, 'CO': co}
            rst = {self.define['argsOut'][k][0]['name']: rst[k] for k in self.define['argsOut']}
            return rst
        else:
            return self.applyGen(state, Logic.BasicP)
    def eval(self, state):
        return self.applyGen(state, Logic.BasicA)
    def z3Interface(self, state):
        return self.applyGen(state, Logic.BasicZ3)

import re
rParseName = r"\.(?P<G>[a-mo-zA-MO-Z]+)(?P<Id>\d*)(?P<N>N){0,1}\((?P<Src>.+?)\)"

class VerilogParser:
    """
        "Verilog" Parser for logic cells
    """
    def __init__(self):
        self.allFunc = set()

    def parseLine(self, l):
        if not l:
            return 
        l = l.replace('((', '( (').replace('))', ') )')
        words = [x.strip() for x in re.split('\s|,|\*', l) if x.strip()]
        if (not words):
            return 
        if (words[-1].endswith(';')):
            lst = words.pop()
            lst = lst[:-1]
            words.append(lst)
            words.append(';')
        curObject = None
        for w in words:
            if (self.state == 'free'):
                if (w == 'module'):
                    return # skip
                if (w == 'endmodule'):
                    self.state = 'over'
                    return # over
                if (w == 'input'):
                    self.state = 'input'
                    continue
                if (w == 'output'):
                    self.state = 'output'
                    continue
                if (w == 'wire'):
                    return # skip
                
                self.curObject = {'func': w.split('_')[0], 'name':'unknown', 'argsIn':{}, 'argsOut':{}}
                self.allFunc.add(self.curObject['func'])
                self.state = 'parseLogic0'
                continue
            if self.state == 'parseLogic0':
                self.curObject['name'] = w
                self.state = 'parseLogic1'
                continue
            if self.state == 'parseLogic1':
                if (w == '('):
                    self.state = 'parseLogic2'
                    continue
            if self.state == 'parseLogic3':
                if self.matches:
                    g = self.matches.group('G')
                    sid = self.matches.group('Id') or 0
                    nn = not not self.matches.group('N')
                    src = self.matches.group('Src') or ''
                    obj = {'group': g, 'id': int(sid), 'name': src, 'inv': nn, 'type': portType(g, self.curObject['func'])}
                    typeName = 'argsIn' if obj['type'] == 'input' else 'argsOut'
                    if (not g in self.curObject[typeName]):
                        self.curObject[typeName][g] = []
                    self.curObject[typeName][g].append(obj)
                    self.wire.add(src)
                    if not (src in self.wireIn):
                        self.wireIn[src] = set()
                    if (obj['type'] == 'input'):
                        self.wireIn[src].add(self.curObject['name'])
                    else:
                        self.wireOut[src] = self.curObject['name']
                    self.state = 'parseLogic2' # fallback to 2
                else:
                    self.matched += w
                    self.matches = re.match(rParseName, self.matched)
                    continue # add and match
            if self.state == 'parseLogic2':
                if (w == ')'):
                    self.state = 'free'
                    self.ops.append(self.curObject)
                    self.curObject = None
                    return # skip
                else:
                    self.matched = w
                    # (:G)(:Id)(:N)\(Src\)
                    self.matches = re.match(rParseName, self.matched)
                    self.state = 'parseLogic3'
                continue
            if self.state == 'input':
                if (w.startswith('keyinput')):
                    self.keyInput.add(w)
                else:
                    self.input.add(w)
                continue
            if self.state == 'output':
                self.output.add(w)
                continue
        self.state = 'free'

    def parseVerilog(self, lines):
        self.ops = []
        self.input = set()
        self.output = set()
        self.keyInput = set()
        self.wire = set()
        self.wireOut = {}
        self.wireIn = {}
        lines = [l.strip() for l in lines]
        lines = [l.split('//', 1)[0].strip() for l in lines]
        lines = [l for l in lines if l]
        lines = "".join(lines).split(';')
        for l in lines:
            self.state = 'free'
            self.parseLine(l)
        self.ops = [Logic(i) for i in self.ops]
        self.opMap = {i.name: i for i in self.ops}
        # print(self.opMap['U3454'])
if __name__ == "__main__":
    ### test only !!!
    import os
    import glob
    allFuncs = set()
    testl1 = Logic({'name': 'U6708', 'argsOut': {'Y': [{'name': 'U3453', 'inv': False, 'group': 'Y', 'type': 'output', 'id': 0}]}, 'argsIn': {'A': [{'name': 'STATE2_REG_0__SCAN_IN', 'inv': False, 'group': 'A', 'type': 'input', 'id': 0}, {'name': 'n6217', 'inv': False, 'group': 'A', 'type': 'input', 'id': 1}, {'name': 'n5577', 'inv': False, 'group': 'A', 'type': 'input', 'id': 2}], 'B': [{'name': 'n7072', 'inv': False, 'group': 'B', 'type': 'input', 'id': 0}, {'name': 'n6217', 'inv': False, 'group': 'B', 'type': 'input', 'id': 1}]}, 'func': 'AOI32'})
    print(str(testl1))
    for filename in glob.glob(os.path.join(os.getcwd(), 'verilog', '*.v')):
        print('processing ' + filename )
        f = open(filename)
        parser = VerilogParser()
        parser.parseVerilog(f.readlines())
        print(' > input size: ' + str(len(parser.input)))
        print(' > output size: ' + str(len(parser.output)))
        print(' > key size: ' + str(len(parser.keyInput)))
        print(' > ops size: ' + str(len(parser.ops)))
        print(' > wire size: ' + str(len(parser.wire)))
        allFuncs = allFuncs | parser.allFunc
        for x in parser.ops:
            if x.func + x.para == 'INV':
                print(str(x))
                import sys
                sys.exit(0)
            # test func
        f.close()
    for f in allFuncs:
        print(f)
#{'NOR3', 'NOR2', 'ADDH', 'ADDF', 'NAND4', 'INV'}
