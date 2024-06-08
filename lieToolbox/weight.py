"""This file stores class Weight and class Weyl group element.
TODO
    1. collingwoods orbit representation and orbit order
    2. more weight and weyl group operations
"""

from copy import deepcopy
from re import split
from math import ceil

# # For flask
# from lieToolbox import RSAlgorithm as rsa
# from lieToolbox import HAlgorithm as ha
# import lieToolbox.DRS_algorithm as drsa

# For test
import RSAlgorithm as rsa
import HAlgorithm as ha
import DRS_algorithm as drsa


"""
Tol = 1e-7
"""

class Weight:
    """This class combines a weight constructed by an entry along with its Lie type. 
    If no entry is given, it will initialize with an empty list with type B. 
    It also support 
    """

    def __init__(lbd, entry: list = [], lieType: str = 'B', type: str = 'R'):
        lbd.entry = entry
        lbd.lieType = lieType
        lbd.type = type
        lbd.weightType = lbd.getWeightType()
        # Comlex case
        lbd.realEntry = lbd.entry
        lbd.imagEntry = []
        if lbd.type == 'C':
            n = int(len(lbd.entry)/2)
            lbd.realEntry = lbd.entry[:n]
            lbd.imagEntry = lbd.entry[n:]
        else:
            n = len(lbd.entry)
        lbd.n = n

    def __getitem__(lbd, key):
        return lbd.entry[key]

    def __setitem__(lbd, key, value):
        lbd.entry[key] = value

    def __add__(lbd, other):
        """
        This function overloads add method to achieve weight operation.

        Args:
            other (Weight): weight object

        Returns:
            Weight: add result
        """
        s = []
        for i in range(len(lbd.entry)):
            s.append(lbd[i] + other[i])
        return Weight(s, lbd.lieType)

    def getWeightType(lbd):
        """This function automatic determines the type of a weight.

        Returns:
            String: Empty, Integral, Half integral, Congruent or mixed
        """
        if len(lbd.entry) == 0:
            return 'Empty'
        elif all(lbd.getEntryType(_) == 'Integer' for _ in lbd.entry):
            return 'Integral'
        elif all(lbd.getEntryType(_) == 'Half Integer' for _ in lbd.entry):
            return 'Half integral'
        elif all(
                lbd.getEntryType(_ - lbd.entry[0]) == 'Integer'
                or lbd.getEntryType(_ + lbd.entry[0]) == 'Integer'
                for _ in lbd.entry):
            return 'Congruent'
        else:
            return 'Mixed'

    def show(lbd):
        """This function shows the weight itself.
        """
        if lbd.type == 'R':
            print(lbd.entry, 'Weight of type', lbd.lieType)
        else:
            entry = []
            for i in range(len(lbd.realEntry)):
                entry.append(complex(lbd.realEntry[i], lbd.imagEntry[i]))
            print(entry, 'Weight of type', lbd.lieType)
            
    def toStr(lbd):
        if len(lbd.entry) == 0:
            entryStr = 'None'
        else:
            if lbd.type == 'C':
                elet = []
                for i in range(len(lbd.realEntry)):
                    if lbd.imagEntry[i] == 0:
                        ele = str(lbd.realEntry[i])
                    elif lbd.realEntry[i] == 0:
                        ele = str(lbd.imagEntry[i]) + 'i'
                    else:
                        ele = str(lbd.realEntry[i]) + '+' + str(lbd.imagEntry[i]) + 'i'
                    elet.append(ele)
                entryStr = '(' + ', '.join(elet) + ')'
            elif lbd.type == 'R':
                entryStr = tuple(lbd.entry)
        return entryStr

    def rightMinus(lbd):
        """This function handles the preparation of a weight element, which
        adds the minus part in the right, i.e. namely x^-.

        Args:
            lbd (Weight): any weight object
        """
        for w in lbd.entry[::-1]:
            lbd.entry.append(-w)

    def leftMinus(lbd):
        """This function handles the preparation of a weyl group element,
        which adds the minus part in the left, namely ^-x.
        """
        minusPart = []
        for w in lbd.entry[::-1]:
            minusPart.append(-w)
        lbd.entry = minusPart + lbd.entry

    def tilde(lbd):
        """This function handles the sequence in \lambda_3, which
        finds its maximum subset and form a congruent list, namely
        \\tilde{x}
        """
        subset = []
        for i in range(len(lbd.entry) - 1, -1, -1):
            if lbd.getEntryType(lbd.entry[i] - lbd.entry[0]) == 'Integer':
                subset.append(lbd.entry[i])
                lbd.entry.pop(i)
        subset.reverse()
        # Justify maximum subset
        if len(subset) >= len(lbd.entry):
            maxSubset = subset
        else:
            maxSubset = lbd.entry
            lbd.entry = subset

        for w in lbd.entry[::-1]:
            maxSubset.append(-w)
        lbd.entry = maxSubset

    def qNegtive(lbd):
        """This function return the q-negtive index for A-type weight.

        Returns:
            int: q negative part
        """
        subset = []
        newlbdEntry = deepcopy(lbd.realEntry)
        for i in range(len(lbd.realEntry) - 1, -1, -1):
            if Weight.getEntryType(newlbdEntry[i] -
                                   newlbdEntry[0]) == 'Integer':
                subset.append(newlbdEntry[i])
                newlbdEntry.pop(i)
        subset.reverse()
        return min(len(subset), len(newlbdEntry))

    def basicDecomposition(lbd):
        """This function decomposes a weight string for Lie type A, which
        returns all the congruent parts.

        Returns:
            list: a list of Weight object
        """
        if lbd.type == 'R':
            newEntry = deepcopy(lbd.entry)
            entryList = []
            while len(newEntry) > 0:
                t = []
                for i in range(len(newEntry) - 1, -1, -1):
                    if lbd.getEntryType(newEntry[i] - newEntry[0]) == 'Integer':
                        t.append(newEntry[i])
                        newEntry.pop(i)
                t.reverse()
                entryList.append(t)
            return [Weight(entry, 'A') for entry in entryList]
        else:
            realEntry = deepcopy(lbd.realEntry)
            imagEntry = deepcopy(lbd.imagEntry)
            weightList = []
            while len(realEntry) > 0:
                realt = []
                imagt = []
                for i in range(len(realEntry) - 1, -1, -1):
                    if lbd.getEntryType(realEntry[i] - realEntry[0]) == 'Integer' and abs(imagEntry[i] - imagEntry[0]) < 1e-7:
                        realt.append(realEntry[i])
                        imagt.append(imagEntry[i])
                        realEntry.pop(i)
                        imagEntry.pop(i)
                realt.reverse()
                imagt.reverse()
                weightList.append(Weight(realt + imagt, 'A', 'C'))
            return weightList
        
        

    def decomposition(lbd):
        """This function decomposes a weight string for Lie type B, C or D to
        integral, half integral and rest part.

        Returns:
            WeightStruct: an object to store decomposed weight data.
        """
        lieType = lbd.lieType
        # Real case
        if lbd.type == 'R':
            newEntry = deepcopy(lbd.entry)
            entry1 = []
            entry2 = []            
            for i in range(len(newEntry) - 1, -1, -1):  # pop from the end
                if lbd.getEntryType(newEntry[i]) == 'Integer':
                    entry1.append(newEntry[i])
                    newEntry.pop(i)
                elif lbd.getEntryType(newEntry[i]) == 'Half integer':
                    entry2.append(newEntry[i])
                    newEntry.pop(i)
            entry1.reverse()  # reverse the list due to previous order
            entry2.reverse()
            lbd1 = Weight(entry1, lieType)
            lbd2 = Weight(entry2, lieType)
            
            lbdList3 = []
            while len(newEntry) > 0:
                entry3 = []
                for i in range(len(newEntry) - 1, -1, -1):  # pop from the end
                    if lbd.getEntryType(
                            newEntry[i] - newEntry[0]) == 'Integer' or lbd.getEntryType(
                                newEntry[i] + newEntry[0]) == 'Integer':
                        entry3.append(newEntry[i])
                        newEntry.pop(i)
                entry3.reverse()
                lbdList3.append(Weight(entry3, 'A', 'R'))
            return WeightStruct(lbd1, lbd2, lbdList3)
        
        # Complex case
        else:
            realEntry = deepcopy(lbd.realEntry)
            imagEntry = deepcopy(lbd.imagEntry)
            entry1 = []
            entry2 = []
            for i in range(len(realEntry) - 1, -1, -1):
                if lbd.getEntryType(realEntry[i]) == 'Integer' and imagEntry[i] == 0:
                    entry1.append(realEntry[i])
                    realEntry.pop(i)
                    imagEntry.pop(i)
                elif lbd.getEntryType(realEntry[i]) == 'Half integer' and imagEntry[i] == 0:
                    entry2.append(realEntry[i])
                    realEntry.pop(i)
                    imagEntry.pop(i)
            entry1.reverse()
            entry2.reverse()
            lbd1 = Weight(entry1, lieType, 'R')
            lbd2 = Weight(entry2, lieType, 'R')
            
            lbdList3 = []
            while len(realEntry) > 0:
                realEntry3 = []
                imagEntry3 = []
                for i in range(len(realEntry) - 1, -1, -1):
                    if (lbd.getEntryType(realEntry[i] - realEntry[0]) == 'Integer' or lbd.getEntryType(
                        realEntry[i] + realEntry[0]) == 'Integer') and abs(imagEntry[i] - imagEntry[0]) < 1e-7:
                        realEntry3.append(realEntry[i])
                        imagEntry3.append(imagEntry[i])
                        realEntry.pop(i)
                        imagEntry.pop(i)
                realEntry3.reverse()
                imagEntry3.reverse()
                lbdList3.append(Weight(realEntry3+imagEntry3, 'A', 'C'))
            
            return WeightStruct(lbd1, lbd2, lbdList3)
                

    def constructTableau(lbd):
        """This function constructs a tableau using Robinson-Schensted algorithm.

        Returns:
            Partition: the shape of tableau is stored in partition
        """
        if lbd.getWeightType(
        ) == 'Empty':  # return empty, if an empty weight was given
            return []
        else:
            if lbd.type == 'R':
                newEntry = deepcopy(lbd.entry)
            else:
                newEntry = deepcopy(lbd.realEntry)
            
            newEntryIndex = rsa.convert2Index(newEntry)
            tableau = rsa.constructYoungTableau(newEntryIndex)
            return tableau
        
    def constructPartition(lbd):
        if lbd.getWeightType(
        ) == 'Empty':  # return empty, if an empty weight was given
            return Partition([], lbd.lieType)
        else:
            tableau = lbd.constructTableau()
            ptEntry = [len(_) for _ in tableau]
            return Partition(ptEntry, lbd.lieType)

    @staticmethod
    def positiveEntry(entry):
        """This function returns the number of positive elements in a sequence.

        Args:
            lbd (Weight): weight object

        Returns:
            int: the number of positive elements
        """
        count = 0
        for k in entry:
            if k > 0:
                count += 1
        return count

    def getAntidominant(lbd):
        """This function calculate the antidominant weight for a given weight.

        Returns:
            Weight: the antidominant weight
        """
        newlbd = deepcopy(lbd)
        n = len(newlbd.entry)
        while newlbd.positiveEntry(newlbd.entry) > 0:
            max_idx = newlbd.entry.index(max(newlbd.entry))
            newlbd[max_idx] = -abs(newlbd[max_idx])
        newlbd.entry.sort()
        return newlbd

    def veryEvenOrbitType(lbd):
        """This function decide the very even orbit for a given weight.

        Returns:
            Str: I or II
        """
        if lbd.lieType == 'A':
            q = lbd.qNegtive()
            if q % 2 == 0:
                return 'I'
            elif q % 2 == 1:
                return 'II'
        else:
            mu = lbd.getAntidominant()
            wg = Weight.getWeylGroupElement(lbd, mu)
            domino = drsa.w2DominoTableau(wg.entry)
            if drsa.vertical_domino_boxes(domino) % 4 == 0:  # DRS algorithm
                return 'I'
            elif drsa.vertical_domino_boxes(domino) % 4 == 2:
                return 'II'
            else:
                return 'None'
    
    def veryEvenOrbitTypeInfo(lbd):
        if lbd.lieType == 'A':
            q = lbd.qNegtive()
            if q % 2 == 0:
                veryEvenType = 'I'
            elif q % 2 == 1:
                veryEvenType = 'II'
            veryEvenTypeInfo = {'Weight': lbd.entry,
                                'qNegtive': q,
                                'VeryEvenType': veryEvenType}
        else:
            mu = lbd.getAntidominant()
            wg = Weight.getWeylGroupElement(lbd, mu)
            domino = drsa.w2DominoTableau(wg.entry)
            verboxNum = drsa.vertical_domino_boxes(domino)
            if verboxNum % 4 == 0:
                veryEvenType = 'I'
            elif verboxNum % 4 == 2:
                veryEvenType = 'II'
            else:
                veryEvenType = 'None'
            
            veryEvenTypeInfo = {'Weight': lbd.entry,
                                'AntidominantWeight': mu.entry,
                                'WeylGroupElement': wg.entry,
                                'DominoTableau': domino,
                                'VerticalBoxNum': verboxNum,
                                'VeryEvenType': veryEvenType}
        return veryEvenTypeInfo


    @staticmethod
    def getWeylGroupElement(lbd, other):
        """This function returns the permutation (Weyl group element) of
        two weights (usually original weight and its corresponding 
        antidominant weight)

        Args:
            other (Weight): weight object

        Returns:
            WeylGroupElement: the required permutation
        """
        n = len(lbd.entry)
        newlbdo = deepcopy(lbd)
        newlbdo.rightMinus()
        newlbd = newlbdo.entry
        newmuo = deepcopy(other)
        newmuo.rightMinus()
        newmu = newmuo.entry
        we = []
        for i in range(2 * n - 1, n - 1, -1):
            for j in range(2 * n - 1, -1, -1):
                if newmu[i] == newlbd[j]:
                    if j < n:
                        entry = j - n
                    elif j >= n:
                        entry = j - n + 1
                    we.append(entry)
                    newlbd[j] = ''
                    break
        we.reverse()
        return WeylGroupElement(we, lbd.lieType)

    @staticmethod
    def getEntryType(x: float):
        """This function decide the element type in an entry

        Args:
            x (element): an element in an entry

        Returns:
            str: type of the element
        """
        if isinstance(x, complex):
            return 'complex'
        if abs(x - round(x)) < 1e-7:
            return 'Integer'
        elif abs(x - int(x) - 0.5) < 1e-7 or abs(x - int(x) + 0.5) < 1e-7:
            return 'Half integer'
        else:
            return 'Not Half integer'
        
    @staticmethod
    def parseStrWeight(input_str: str, lieType: str):
        # Complex
        if 'i' in input_str:
            ntype = 'C'
            parts = split(', |,|，| ', input_str)
            real_parts = []
            imag_parts = []
            
            for part in parts:
                print('s', part)
                if 'i' in part:
                    # Ordinary complex number
                    if '+' in part:
                        real, imag = part.split('+')
                        print(real, imag)
                        real_parts.append(eval(real) if real else 0)
                        if imag == 'i':
                            imag_parts.append(1)
                        else:
                            imag_parts.append(eval(imag.replace('i', '').strip()))
                    # Pure imaginary number
                    else:
                        real_parts.append(0)
                        if part == 'i':
                            imag_parts.append(1)
                        else:
                            imag_parts.append(eval(part.replace('i', '').strip()))
                # Real number
                else:
                    real_parts.append(eval(part))
                    imag_parts.append(0)

            entry = real_parts + imag_parts
        
        # Real  
        else:
            ntype = 'R'
            entry = list(map(eval, split(', |,|，| ', input_str)))
        
        return Weight(entry=entry, lieType=lieType, type=ntype)


class WeightStruct:
    """This class stores information of a decomposed weight.
    """

    def __init__(lbdStruct,
                 lbd1: Weight,
                 lbd2: Weight,
                 lbdList3: list[Weight] = []):
        WeightStruct.Integral = lbd1
        WeightStruct.HIntegral = lbd2
        WeightStruct.NHIntegral = lbdList3
        WeightStruct.lieType = lbd1.lieType
        if len(lbdList3):
            WeightStruct.type = lbdList3[0].type
        else:
            WeightStruct.type = 'R'

    def show(lbdStruct):
        print('Weight struct of type', lbdStruct.lieType)
        print('\t Integral part:', lbdStruct.Integral.entry)
        print('\t Half integral part:', lbdStruct.HIntegral.entry)
        print('\t Rest part:')
        if len(WeightStruct.NHIntegral) == 0:
            print('None')
        else:
            for lbd in WeightStruct.NHIntegral:
                print('\t\t ',lbd.entry)

    def serialize(lbdStruct):
        lbdStructDict = {}
        lbdStructDict['Integral'] = lbdStruct.Integral.entry
        lbdStructDict['HIntegral'] = lbdStruct.HIntegral.entry
        lbdStructDict['NHIntegral'] = [_.entry for _ in lbdStruct.NHIntegral]
        return lbdStructDict
        

    def veryEvenOrbitType(lbdStruct):
        """This function is the main method to decide the very even orbit type
        for Lie type D. It uses different ways to decide whether integral and 
        half integral part is empty.

        Returns:
            str: very even orbit type
        """
        veryEvenType1 = 'None'  # type combined by int part
        veryEvenType2 = 'None'  # type combined by half-int part
        veryEvenType3 = 'None'  # type contributed by rest part
        
        
        # Very even type contributed by int and half-int part
        if len(lbdStruct.Integral.entry) != 0:
            veryEvenType1 = lbdStruct.Integral.veryEvenOrbitType()
        
        if len(lbdStruct.HIntegral.entry) != 0:
            veryEvenType2 = lbdStruct.HIntegral.veryEvenOrbitType()
        
        # Very even part contributed by the rest part
        if len(lbdStruct.NHIntegral) != 0:
            q = 0
            for lbdk in lbdStruct.NHIntegral:
                q += lbdk.qNegtive()
            if q % 2 == 0:
                veryEvenType3 = 'I'
            elif q % 2 == 1:
                veryEvenType3 = 'II'
            # print(lbdStruct.NHIntegral, q, veryEvenType3)
                
        # Combination of 3 types
        if veryEvenType1 == 'None' and veryEvenType2 == 'None':
            veryEvenType = veryEvenType3
        elif veryEvenType1 == 'None' and veryEvenType3 == 'None':
            veryEvenType = veryEvenType2
        elif veryEvenType2 == 'None' and veryEvenType3 == 'None':
            veryEvenType = veryEvenType1
        else:
            if veryEvenType1 == 'None':
                veryEvenType12 = veryEvenType2
            elif veryEvenType2 == 'None':
                veryEvenType12 = veryEvenType1
            else:
                if veryEvenType1 == veryEvenType2:
                    veryEvenType12 = 'I'
                if veryEvenType1 != veryEvenType2:
                    veryEvenType12 = 'II'
            
            if veryEvenType3 == 'None' or veryEvenType3 == 'I':
                veryEvenType = veryEvenType12
            
            elif veryEvenType3 == 'II':
                if veryEvenType12 == 'I':
                    veryEvenType = 'II'
                elif veryEvenType12 == 'II':
                    veryEvenType = 'I'
        
        # print(veryEvenType1,veryEvenType2,veryEvenType12, veryEvenType3)
        return veryEvenType
    
    def veryEvenOrbitTypeInfo(lbdStruct):
        veryEvenTypeInfo1 = 'None'  # type combined by int part
        veryEvenType1 = 'None'
        veryEvenTypeInfo2 = 'None'  # type combined by half-int part
        veryEvenType2 = 'None'
        veryEvenTypeInfoList3 = []  # type contributed by rest part
        veryEvenType3 = 'None'
        
        # Very even type contributed by int and half-int part
        if len(lbdStruct.Integral.entry) != 0:
            veryEvenTypeInfo1 = lbdStruct.Integral.veryEvenOrbitTypeInfo()
            veryEvenType1 = veryEvenTypeInfo1['VeryEvenType']
        
        if len(lbdStruct.HIntegral.entry) != 0:
            veryEvenTypeInfo2 = lbdStruct.HIntegral.veryEvenOrbitTypeInfo()
            veryEvenType2 = veryEvenTypeInfo2['VeryEvenType']
        
        
        # Very even part contributed by the rest part
        if len(lbdStruct.NHIntegral) != 0:
            q = 0
            for lbdk in lbdStruct.NHIntegral:
                q += lbdk.qNegtive()
                veryEvenTypeInfoList3.append(lbdk.veryEvenOrbitTypeInfo())
            if q % 2 == 0:
                veryEvenType3 = 'I'
            elif q % 2 == 1:
                veryEvenType3 = 'II'
            # print(lbdStruct.NHIntegral, q, veryEvenType3)
                
        # Combination of 3 types
        if veryEvenType1 == 'None' and veryEvenType2 == 'None':
            veryEvenType = veryEvenType3
        elif veryEvenType1 == 'None' and veryEvenType3 == 'None':
            veryEvenType = veryEvenType2
        elif veryEvenType2 == 'None' and veryEvenType3 == 'None':
            veryEvenType = veryEvenType1
        else:
            if veryEvenType1 == 'None':
                veryEvenType12 = veryEvenType2
            elif veryEvenType2 == 'None':
                veryEvenType12 = veryEvenType1
            else:
                if veryEvenType1 == veryEvenType2:
                    veryEvenType12 = 'I'
                if veryEvenType1 != veryEvenType2:
                    veryEvenType12 = 'II'
            
            if veryEvenType3 == 'None' or veryEvenType3 == 'I':
                veryEvenType = veryEvenType12
            
            elif veryEvenType3 == 'II':
                if veryEvenType12 == 'I':
                    veryEvenType = 'II'
                elif veryEvenType12 == 'II':
                    veryEvenType = 'I'
        
        # print(veryEvenType12, veryEvenType3)
        
        veryEvenTypeInfo = {'veryEvenType': veryEvenType,
                            'Integral': veryEvenTypeInfo1,
                            'HIntegral': veryEvenTypeInfo2,
                            'NHIntegral': veryEvenTypeInfoList3}
        return veryEvenTypeInfo

class HighestWeightModule:
    def __init__(self, lbd: Weight):
        self.highestWeight = lbd
    
    def nilpotentOrbit(self):
        """This is the main function to classify the nilpotent orbit of highest
        weight module of four types of classical Lie algebra. It also has a strategy
        to handle very even orbit of type D using Domino Robinson Schensted algorithm.

        Returns:
            Partition: combined with very even orbit type of type D
        """
        lbd = self.highestWeight
        if lbd.lieType == 'A':
            lbdList = lbd.basicDecomposition()
            p = Partition([], lbd.lieType)
            for lbdk in lbdList:
                p += lbdk.constructPartition()
            orbit = NilpotentOrbit(p.entry, p.lieType) # create an orbit

        else:
            # Integral part
            lbdStruct = lbd.decomposition()
            lbdStruct.Integral.rightMinus()
            p1 = lbdStruct.Integral.constructPartition()
            p1.hollowBoxAlgorithm(lbd.lieType)
            
            # Half integral part
            lbdStruct.HIntegral.rightMinus()
            p2 = lbdStruct.HIntegral.constructPartition()
            if lbd.lieType == 'B':
                p2.hollowBoxAlgorithm('D')
            elif lbd.lieType == 'C' or 'D':
                p2.hollowBoxAlgorithm('metaplectic')
            # Rest part
            p3 = Partition([], lbd.lieType)
            for lbdk in lbdStruct.NHIntegral:
                lbdk.tilde()
                p3 += lbdk.constructPartition()
            # p1.show()
            # p2.show()
            # lbdStruct.Integral.show()
            # Partition union
            p = p1 + p2 + p3 + p3
            # Collapse to get the orbit
            p.collapse()
            
            orbit = NilpotentOrbit(p.entry, p.lieType)  # create an orbit
            
            # Justify and handle very even orbit
            if lbd.lieType == 'D' and p.isVeryEven() == True:
                orbit.veryEven = True
                orbit.veryEvenType = WeightStruct.veryEvenOrbitType(
                    lbd.decomposition())
        return orbit

    def nilpotentOrbitInfo(self):
        """Calculate the orbit and return detailed information in the calculation process

        Returns:
            Dict: Orbit Information
            Type A: lieType, WeightList, PartitionList, Orbit
            Type B and C: lieType, WeightStruct, PartitionList, Orbit
            Type D: lieType, WeightStruct, PartitionList, Orbit, veryEven, veryEvenTypeInfo
                veryEvenType
        """
        lbd = self.highestWeight
        if lbd.lieType == 'A':
            lbdList = lbd.basicDecomposition()
            pList = []
            p = Partition([], lbd.lieType)
            for lbdk in lbdList:
                pk = lbdk.constructPartition()
                pList.append(pk)
                p += pk
            orbit = NilpotentOrbit(p.entry, p.lieType) # create an orbit
            
            # Orbit Info to Dict
            orbitInfo = {}
            orbitInfo['lieType'] = 'A'
            orbitInfo['highestWeight'] = lbd.toStr()
            orbitInfo['n'] = lbd.n
            orbitInfo['UnitList'] = []
            for i in range(len(pList)):
                orbitInfo['UnitList'].append({'Num': i+1,
                                              'Weight': lbdList[i].toStr(), 
                                              'Partition': pList[i].entry})
            orbitInfo['Orbit'] = orbit.entry
            
            return orbitInfo

        else:
            # Integral part
            lbdStruct = lbd.decomposition()
            lbdStruct.Integral.rightMinus()
            p1 = lbdStruct.Integral.constructPartition()
            p1oEntry = deepcopy(p1.entry) # before H
            p1.hollowBoxAlgorithm(lbd.lieType)
            # Half integral part
            lbdStruct.HIntegral.rightMinus()
            p2 = lbdStruct.HIntegral.constructPartition()
            p2oEntry = deepcopy(p2.entry)
            if lbd.lieType == 'B':
                p2.hollowBoxAlgorithm('D')
            elif lbd.lieType == 'C' or 'D':
                p2.hollowBoxAlgorithm('metaplectic')
            # Rest part
            p3 = Partition([], lbd.lieType)
            p3List = []
            for lbdk in lbdStruct.NHIntegral:
                lbdk.tilde()
                p3k = lbdk.constructPartition()
                p3List.append(p3k)
                p3 += p3k
            # Partition union
            p = p1 + p2 + p3 + p3
            
            # Collapse to get the orbit
            p.collapse()
            orbit = NilpotentOrbit(p.entry, p.lieType)  # create an orbit
            
            orbitInfo = {}
            orbitInfo['lieType'] = lbd.lieType
            orbitInfo['highestWeight'] = lbd.toStr()
            orbitInfo['n'] = lbd.n
            orbitInfo['Integral'] = {'Weight': lbdStruct.Integral.toStr(),
                                     'Partition1': p1oEntry,
                                     'Partition2': p1.entry}
            orbitInfo['HIntegral'] = {'Weight': lbdStruct.HIntegral.toStr(),
                                     'Partition1': p2oEntry,
                                     'Partition2': p2.entry}
            orbitInfo['NHIntegral'] = []
            for i in range(len(p3List)):
                orbitInfo['NHIntegral'].append({'Num': i+1,
                                              'Weight': lbdStruct.NHIntegral[i].toStr(), 
                                              'Partition': p3List[i].entry})
            orbitInfo['Orbit'] = orbit.entry
            # print(orbitInfo)
            
            # Justify and handle very even orbit
            if lbd.lieType == 'D' and p.isVeryEven() == True:
                orbit.veryEven = True
                lbdStructD = lbd.decomposition()
                
                    
                orbit.veryEvenType = WeightStruct.veryEvenOrbitType(
                    lbd.decomposition())
                lbd.show()
                orbitInfo['isVeryEven'] = orbit.veryEven
                orbitInfo['veryEvenTypeInfo'] = WeightStruct.veryEvenOrbitTypeInfo(lbd.decomposition())
                orbitInfo['veryEvenType'] = orbit.veryEvenType
                # print(orbitInfo)
                
                
        return orbitInfo
    
    @staticmethod
    def a_fun(obt: 'Partition', a_fun_type: str):
        a_fun_val = 0
        if not obt.entry:
            return 0
        
        if a_fun_type == 'a':
            for i,ele in enumerate(obt.entry):
                a_fun_val += i*ele
        elif a_fun_type == 'b':
            for i,ele in enumerate(obt.oddEntry()):
                # print(obt.oddEntry())
                a_fun_val += i*ele
        elif a_fun_type == 'd':
            for i,ele in enumerate(obt.evenEntry()):
                a_fun_val += i*ele
        return a_fun_val
    
    def GKdim(self):
        lbd = self.highestWeight
        lieType = lbd.lieType
        n = lbd.n 
        L_lbd = HighestWeightModule(lbd)
        obt = L_lbd.nilpotentOrbit()
        print(L_lbd.nilpotentOrbitInfo())
        obt_info = L_lbd.nilpotentOrbitInfo()
        
        if lieType == 'A':
            partitions = [Partition(d['Partition']) for d in obt_info['UnitList']]
            gk_dim = n*(n-1)/2
            for partition in partitions:
                gk_dim -= HighestWeightModule.a_fun(partition, 'a')
        else:
            integral_partition = Partition(obt_info['Integral']['Partition2'])
            half_integral_partition = Partition(obt_info['HIntegral']['Partition2'])
            non_integral_partitions = [Partition(d['Partition']) for d in obt_info['NHIntegral']]
        
            if lieType == 'B':
                gk_dim = n*n - (HighestWeightModule.a_fun(integral_partition, 'b')
                                + HighestWeightModule.a_fun(half_integral_partition, 'b'))
                for non_integral_partition in non_integral_partitions:
                    gk_dim -= HighestWeightModule.a_fun(non_integral_partition, 'a')
            elif lieType == 'C':
                gk_dim = n*n - (HighestWeightModule.a_fun(integral_partition, 'b')
                                + HighestWeightModule.a_fun(half_integral_partition, 'd'))
                for non_integral_partition in non_integral_partitions:
                    gk_dim -= HighestWeightModule.a_fun(non_integral_partition, 'a')
            elif lieType == 'D':
                gk_dim = n*n - n - (HighestWeightModule.a_fun(integral_partition, 'd')
                                + HighestWeightModule.a_fun(half_integral_partition, 'd'))
                for non_integral_partition in non_integral_partitions:
                    gk_dim -= HighestWeightModule.a_fun(non_integral_partition, 'a')
            # print(n, afun, obt.entry)
        return int(gk_dim)
        
    def GKdimInfo(self):
        lbd = self.highestWeight
        lieType = lbd.lieType
        n = lbd.n
        L_lbd = HighestWeightModule(lbd)
        obt = L_lbd.nilpotentOrbit()
        obt_info = L_lbd.nilpotentOrbitInfo()
        afun = 0
        gkDimInfo = {'Weight': lbd.toStr(), 
                     'lieType': lieType, 
                     'n': n, 
                     'Orbit': obt.entry}
        
        if lieType == 'A':
            partitions = [Partition(d['Partition']) for d in obt_info['UnitList']]
            gk_dim = n*(n-1)/2
            for partition in partitions:
                afun += HighestWeightModule.a_fun(partition, 'a')
            gk_dim -= afun 
        
        else:
            integral_partition = Partition(obt_info['Integral']['Partition2'])
            half_integral_partition = Partition(obt_info['HIntegral']['Partition2'])
            non_integral_partitions = [Partition(d['Partition']) for d in obt_info['NHIntegral']]
        
            if lieType == 'B':
                afun += (HighestWeightModule.a_fun(integral_partition, 'b') + 
                         HighestWeightModule.a_fun(half_integral_partition, 'b'))
                for non_integral_partition in non_integral_partitions:
                    afun += HighestWeightModule.a_fun(non_integral_partition, 'a')
                gk_dim = n*n - afun
                
            elif lieType == 'C':
                afun += (HighestWeightModule.a_fun(integral_partition, 'b') + 
                         HighestWeightModule.a_fun(half_integral_partition, 'd'))
                for non_integral_partition in non_integral_partitions:
                    afun += HighestWeightModule.a_fun(non_integral_partition, 'a')
                gk_dim = n*n - afun
                
            elif lieType == 'D':
                afun += (HighestWeightModule.a_fun(integral_partition, 'd') + 
                         HighestWeightModule.a_fun(half_integral_partition, 'd'))
                for non_integral_partition in non_integral_partitions:
                    afun += HighestWeightModule.a_fun(non_integral_partition, 'a')
                gk_dim = n*n - n - afun
        
        gkDimInfo['a'] = afun
        gkDimInfo['gkdim'] = gk_dim
        return gkDimInfo
        
        

class WeylGroupElement:
    """This class stores data of an element in the weyl group, which can be
    expressed as a signed permutation. It also support the multiple operation.
    """

    def __init__(wg, entry: list = [], lieType: str = 'B'):
        wg.entry = entry
        wg.lieType = lieType

    def __getitem__(wg, key):
        return wg.entry[key]

    def action(wg, lbd: Weight):
        """This function is a default group action on a weight (linear combination
        of a root system).

        Args:
            lbd (Weight): weight object

        Returns:
            Weight: new weight object
        """
        newlbd = Weight([], )
        newlbd.lieType = lbd.lieType
        for wgk in wg:
            if wgk > 0:
                newlbd.entry.append(lbd[wgk - 1])
            elif wgk < 0:
                newlbd.entry.append(-lbd[-wgk - 1])
        return newlbd

    def __mul__(wg, other):
        """This function is a default group multiplication.

        Args:
            other (WeylGroupElement): Weyl group element

        Returns:
            WeylGroupElement: Weyl group element
        """
        lbde = Weight(list(range(1, len(wg.entry) + 1)), wg.lieType)
        lbd1 = other.action(lbde)
        lbdr = wg.action(lbd1)
        return WeylGroupElement(lbdr.entry, wg.lieType)

    def leftMinus(wg):
        """This function handles the preparation of a weyl group element,
        which adds the minus part in the left, namely ^-x.
        """
        minusPart = []
        for w in wg.entry[::-1]:
            minusPart.append(-w)
        wg.entry = minusPart + wg.entry

    def constructPartition(wg):
        """This function construct the tableau using Robinson-Schensted Algorithm.

        Returns:
            Partition: the tableau shape
        """
        if len(wg.entry) == 0:  # return empty, if an empty weight was given
            return Partition([], wg.lieType)
        else:
            newEntry = deepcopy(wg.entry)
            newEntryIndex = rsa.convert2Index(newEntry)
            Tableau = rsa.constructYoungTableau(newEntryIndex)
            ptEntry = [len(_) for _ in Tableau]
            return Partition(ptEntry, wg.lieType)

    def show(wg):
        print(wg.entry, 'lieType:', wg.lieType)


class Partition:
    """This class stores either the infomation of a Young Tableau or the
    representation of nilpotent orbit.
    """

    def __init__(pt, entry: list = [], lieType: str = 'B'):
        pt.entry = entry
        pt.lieType = lieType

    def __add__(pt, other):
        """This function overloads '+' to obtain the union of two partitions.

        Args:
            other (Partition): Partition object

        Returns:
            Partition: Partition object
        """
        pu = []
        p_1 = deepcopy(pt.entry)
        p_2 = deepcopy(other.entry)
        if len(p_1) <= len(p_2):  # fill zeros to achieve same length
            p_1 += (len(p_2) - len(p_1)) * [0]
        else:
            p_2 += (len(p_1) - len(p_2)) * [0]
        for i in range(len(p_1)):
            pu.append(p_1[i] + p_2[i])
        return Partition(pu, pt.lieType)
    
    def __eq__(pt, other):
        """This function compares two partitions (resp. Young Tableau)

        Args:
            other (Partition): Partition object
        
        Returns:
            Bool
        """
        p_1 = pt.entry
        p_2 = other.entry
        while p_1[-1] == 0:
            p_1.pop()
        while p_2[-1] == 0:
            p_2.pop()
        return p_1 == p_2

    def show(pt):
        """This function shows partition itself, which also checks whether
        the partition is special.
        """
        if len(pt.entry) == 0:
            print('Empty')
        else:
            print(pt.entry)

    def isSpecialType(pt):
        """This function checks whether the partition is special, the notation
        is stated as follows:
        Rules for nilpotent orbits
            A_n: Any P(n)
            B_n: so(2n+1), P(1,2n+1), even parts occur with even multiplicity
            C_n: sp(2n), P(-1,2n), odd parts occur with even multiplicity
            D_n: so(2n), P(1,2n), even parts occur with even multiplicity
                except that "very even" partitions -> Type I and Type II

        Returns:
            bool: True or False
        """
        while pt.entry[-1] == 0:  # delete zeros
            pt.entry.pop()
        flag = True

        if pt.lieType == 'A':
            flag = True

        elif pt.lieType == 'B':
            if sum(pt.entry) % 2 == 1:  # check sum
                for p_k in pt.entry:
                    if p_k % 2 == 0:  # B rule
                        if pt.entry.count(p_k) % 2 != 0:
                            flag = False
                            break
            else:
                flag = False

        elif pt.lieType == 'C':
            if sum(pt.entry) % 2 == 0:  # check sum
                for p_k in pt.entry:
                    if p_k % 2 == 1:  # C rule
                        if pt.entry.count(p_k) % 2 != 0:
                            flag = False
                            break
            else:
                flag = False

        elif pt.lieType == 'D':
            if sum(pt.entry) % 2 == 0:  # check sum
                for p_k in pt.entry:
                    if p_k % 2 == 0:  # D rule
                        if pt.entry.count(p_k) % 2 != 0:
                            flag = False
                            break
            else:
                flag = False

        return flag

    def isVeryEven(pt):
        """This function checks whether a partition of type D is very
        even.

        Returns:
            bool: True or False
        """
        if pt.lieType == 'D':
            newp = deepcopy(pt.entry)
            while newp[-1] == 0:  # delete zeros
                newp.pop()
            flag = 0
            for p_k in newp:
                if p_k % 2 == 0 and newp.count(p_k) % 2 == 0:
                    flag += 1
            if flag == len(newp):
                return True
            else:
                return False
        else:
            return False

    def collapse(pt):
        """This function carries out standard collapse operation introduced in
        CM93 to transfer the partition to correct type.
        """
        if pt.lieType == 'A':
            pass
        elif pt.lieType == 'B':
            if sum(pt.entry) % 2 == 0:  # convert to P(2n+1) for type B
                pt.entry[0] += 1
            # pt.show()
            while pt.isSpecialType() == False:
                pt.entry.append(0)
                for i in range(len(pt.entry)):
                    # even
                    if pt.entry[i] % 2 == 0 and pt.entry.count(
                            pt.entry[i]) % 2 == 1:
                        pt.entry[i] -= 1
                        for j in range(i, len(pt.entry)):
                            if pt.entry[j] < pt.entry[i]:
                                pt.entry[j] += 1
                                break
                        break

        elif pt.lieType == 'C':
            while pt.isSpecialType() == 0:
                pt.entry.append(0)
                for i in range(len(pt.entry)):
                    # odd
                    if pt.entry[i] % 2 == 1 and pt.entry.count(
                            pt.entry[i]) % 2 == 1:
                        pt.entry[i] -= 1
                        for j in range(i, len(pt.entry)):
                            if pt.entry[j] < pt.entry[i]:
                                pt.entry[j] += 1
                                break
                        break

        elif pt.lieType == 'D':
            while pt.isSpecialType() == 0:
                pt.entry.append(0)
                for i in range(len(pt.entry)):
                    # even
                    if pt.entry[i] % 2 == 0 and pt.entry.count(
                            pt.entry[i]) % 2 == 1:
                        pt.entry[i] -= 1
                        for j in range(i, len(pt.entry)):
                            if pt.entry[j] < pt.entry[i]:
                                pt.entry[j] += 1
                                break
                        break

        while pt.entry[-1] == 0:  # delete zeros
            pt.entry.pop()

    def restrictedCollapse(pt):
        pass

    def expansion(pt):
        """This function carries out standard expansion operation introduced in
        CM93 to transfer the partition to correct type.
        """
        if pt.lieType == 'A':
            pass
        
        elif pt.lieType == 'B':
            newp = deepcopy(pt.entry)
            for i in range(0, len(pt.entry), 2):
                if pt.entry[i]%2 == 0 and pt.entry[i+1] == pt.entry[i] and (i == 0 or pt.entry[i-1]!=pt.entry[i]):
                    newp[i] += 1
                    newp[i+1] -= 1
            newp.sort(reverse=True)
            pt.entry = newp
            while pt.entry[-1] == 0:
                pt.entry.pop()
        
        elif pt.lieType == 'C':
            for i in range(1, len(pt.entry), 2):
            # print(i)
                if pt.entry[i]%2 == 1 and i+1 < len(pt.entry) and pt.entry[i+1] == pt.entry[i] and pt.entry[i-1] != pt.entry[i]:
                    newp[i] += 1
                    newp[i+1] -= 1
            newp.sort(reverse=True)
            pt.entry = newp
            while pt.entry[-1] == 0:
                pt.entry.pop()
        
        elif pt.lieType == 'D':
            newp = deepcopy(pt.entry)
            for i in range(1, len(pt.entry), 2):
                if pt.entry[i]%2 == 0 and i+1 < len(pt.entry) and pt.entry[i+1] == pt.entry[i] and pt.entry[i-1]!=pt.entry[i]:
                    newp[i] += 1
                    newp[i+1] -= 1
            newp.sort(reverse=True)
            pt.entry = newp
            while pt.entry[-1] == 0:
                pt.entry.pop()
                
    def oddEntry(pt):
        p = pt.entry
        p_even = []
        for i in range(len(p)):
            if i%2 == 0:
                p_even.append(int(p[i]/2))
            else:
                p_even.append(ceil(p[i]/2))
        while p_even[-1] == 0:
            p_even.pop()
        return p_even
    
    def evenEntry(pt):
        p = pt.entry
        p_odd = []
        for i in range(len(p)):
            if i%2 == 0:
                p_odd.append(ceil(p[i]/2))
            else:
                p_odd.append(int(p[i]/2))
        while p_odd[-1] == 0:
            p_odd.pop()
        return p_odd
    
    def evenPartitionFrame(pt):
        """Get even partition
        """
        if len(pt.entry) == 0:
            p_even = []
        else:
            p_even = []
            for i in range(len(pt.entry)):
                if (pt.entry[i] + i + 1) % 2 == 1:
                    p_even.append(pt.entry[i] - 1)
                else:
                    p_even.append(pt.entry[i])
        return Partition(p_even, pt.lieType)
    
    def oddPartitionFrame(pt):
        """Get odd partition
        """
        if len(pt.entry) == 0:
            p_odd = []
        else:
            p_odd = []
            for i in range(len(pt.entry)):
                if (pt.entry[i] + i + 1) % 2 == 0:
                    p_odd.append(pt.entry[i] - 1)
                else:
                    p_odd.append(pt.entry[i])
        return Partition(p_odd, pt.lieType)

    def hollowBoxAlgorithm(pt, lieType):
        """This function carries out H-algorithm.

        Args:
            lieType (str): Lie type
        """
        if len(pt.entry) == 0:
            pass  # do nothing if empty
        else:
            if lieType == 'A':
                pt.entry = ha.H_algorithm(pt.entry, 1)
            elif lieType == 'B':
                pt.entry = ha.H_algorithm(pt.entry, 2)
            elif lieType == 'C':
                pt.entry = ha.H_algorithm(pt.entry, 3)
            elif lieType == 'D':
                pt.entry = ha.H_algorithm(pt.entry, 4)
            elif lieType == 'metaplectic':
                pt.entry = ha.H_algorithm(pt.entry, 5)
    
    def hollowBoxLabel(pt, lieType):
        if len(pt.entry) == 0:
            label = [0]  # do nothing if empty
        else:
            if lieType == 'A':
                label = ha.getLabel(pt.entry, 1)
            elif lieType == 'B':
                label = ha.getLabel(pt.entry, 2)
            elif lieType == 'C':
                label = ha.getLabel(pt.entry, 3)
            elif lieType == 'D':
                label = ha.getLabel(pt.entry, 4)
            elif lieType == 'metaplectic':
                label = ha.getLabel(pt.entry, 5)
        return label

    def convert2Symbol(pt):
        """This function constructs a Lusztig Symbol through partition.

        Returns:
            Symbol: a B-Symbol or D-Symbol
        """
        p = deepcopy(pt.entry)
        if len(p) % 2 == 0:
            p += [0]
        s = [p[len(p) - i - 1] + i for i in range(len(p))]
        bs_even = [int(i / 2) for i in s if i % 2 == 0]
        bs_odd = [int((i - 1) / 2) for i in s if i % 2 != 0]
        if pt.lieType == 'B' or pt.lieType == 'C':
            return Symbol(bs_even, bs_odd, pt.lieType)
        elif pt.lieType == 'D':
            ds_even = bs_even
            ds_odd = [0] + [i + 1 for i in bs_odd]
            return Symbol(ds_even, ds_odd, 'D')


class Symbol:
    """This class handles structure and operation of Lusztig Symbol.
    """

    def __init__(ls,
                 topEntry: list = [],
                 bottomEntry: list = [],
                 lieType: str = 'B'):
        Symbol.topEntry = topEntry
        Symbol.bottomEntry = bottomEntry
        Symbol.lieType = lieType
        Symbol.special = False

    def show(ls):
        """This function shows the Symbol itself.
        """
        print(ls.lieType, 'Symbol')
        print('Top row:', ls.topEntry)
        print('Bottom row', ls.bottomEntry)

    def makeSpecial(ls):
        """This function sorts the top and bottom row to make a special
        Symbol.
        """
        ls_combine = ls.topEntry + ls.bottomEntry
        ls_combine_sp = sorted(ls_combine)
        ls.topEntry = [i for i in ls_combine_sp[::2]]
        ls.bottomEntry = [i for i in ls_combine_sp[1::2]]
        ls.special = True

    def convert2Partition(ls) -> Partition:
        """This function uses Springer correspondance to construct special
        partition from special Symbol.

        Returns:
            Partition: special partition
        """
        if ls.special == False:
            return None
        else:
            if ls.lieType == 'B' or ls.lieType == 'D':
                s_even_sp = [2 * i + 1 for i in ls.topEntry]
                s_odd_sp = [2 * i for i in ls.bottomEntry]
                s_sp = sorted(s_even_sp + s_odd_sp)
                p_sp = [
                    s_sp[len(s_sp) - i - 1] - (len(s_sp) - i - 1)
                    for i in range(len(s_sp))
                ]

            elif ls.lieType == 'C':
                s_even_sp = [2 * i for i in ls.topEntry]
                s_odd_sp = [2 * i + 1 for i in ls.bottomEntry]
                s_sp = sorted(s_even_sp + s_odd_sp)
                p_sp = [
                    s_sp[len(s_sp) - i - 1] - (len(s_sp) - i - 1)
                    for i in range(len(s_sp))
                ]

            while p_sp[-1] == 0:
                p_sp.pop()
            return Partition(p_sp, ls.lieType)


class NilpotentOrbit(Partition):
    def __init__(self, entry: list = ..., lieType: str = 'B'):
        super().__init__(entry, lieType)
        self.veryEven = False
        self.veryEvenType = None
        self.entry.sort(reverse=True)
    
    def show(self) -> None:
        """show the orbit itself
        """
        if self.veryEven == True:
            print(self.entry, self.veryEvenType, 'Nilpotent orbit of type D')
        else:
            print(self.entry, 'Nilpotent orbit of type', self.lieType)
        
    
    def toStr(self):
        """This function returns a string for quick identification of 
        nilpotent orbit type.

        Returns:
            str: e.g. [2, 2, 2, 2] I
        """
        if self.veryEven == False:
            return str(self.entry) + ', Orbit of Type ' + self.lieType
        else:
            return str(self.entry) + " " + str(self.veryEvenType) + ', Orbit of Type D'

    @staticmethod
    def generateOrbitList(n:int, lieType:str) -> list:
        """This function generates all possible equivalent classes of a
        Lie module with dim n using backtrack method

        Args:
            n (int): dimension
            lieType (str): lieType

        Returns:
            list: list of orbits
        """
        if lieType == 'A':
            l = n
        elif lieType == 'B':
            l = 2 * n + 1
        else:
            l = 2 * n
        def backtrack(start, target, path):
            if target == 0:
                pt = NilpotentOrbit(list(path), lieType)
                if pt.isSpecialType():
                    # Very even type
                    if pt.lieType == 'D' and pt.isVeryEven():
                        pt.veryEven = True
                        pt.veryEvenType = 'I'
                        orbitList.append(pt)
                        pt1 = deepcopy(pt)
                        pt1.veryEvenType = 'II'
                        orbitList.append(pt1)
                    else:
                        orbitList.append(pt)
                return 
            for i in range(min(start, target), 0, -1):
                path.append(i)
                backtrack(i, target - i, path)
                path.pop()

        orbitList = []
        backtrack(l, l, [])
        return orbitList
    
    def __gt__(self, other):
        """Partial order of orbits

        Args:
            other (NilpotentOrbit): another orbit

        Returns:
            bool: whether the orbit is higher than the other
        """
        p_1 = self.entry
        p_2 = other.entry
        if len(p_1) <= len(p_2):  # fill zeros to achieve same length
            p_1 += (len(p_2) - len(p_1)) * [0]
        else:
            p_2 += (len(p_1) - len(p_2)) * [0]
        return all([p_1[i] >= p_2[i] for i in range(len(p_1))])


if __name__ == '__main__':
    lbd1 = Weight([1.1, 2, 0.1, 1.5, 4, 2.5,-1, 7,-3, 6,-8, 5], 'D')
    lbd2 = Weight([4,3,-5,6],'D')
    lbd3 = Weight([3,2,1,-5,-6,7], 'D')
    lbd4 = Weight([1.1, 2, 1.5, 4, 2.5,-1, 7,-3, 6,-8, 5, 0.1], 'D')
    lbd5 = Weight([2.1,1.1,-0.1,2.1,2,4,2,0.9], 'B')
    L_lbd1 = HighestWeightModule(lbd1)
    obt1 = L_lbd1.nilpotentOrbit()
    obt1.show()
    obtinfo1 = L_lbd1.nilpotentOrbitInfo()
    print(obtinfo1)
    
    lbd6 = Weight([1,2,1,1,0,1,1,1], 'B','C')
    lbd7 = Weight([4,3,-5,6,1.1,2.1,0.9,1.9,3.1,5.1,1.5,1.5,
                   0,0,0,0,1,1,1,2,2,2,3,3], 'B', 'C')
    print(lbd6.realEntry)
    wtl = lbd6.basicDecomposition()
    print('wtl:')
    for w in wtl:
        w.show()
    
    wts = lbd7.decomposition()
    wts.show()
    obt2 = HighestWeightModule(lbd7).nilpotentOrbit()
    obt2.show()
    
    lbd = Weight([1,7,4,2,5,6], 'B')
    L_lbd = HighestWeightModule(lbd)
    L_lbd.nilpotentOrbit().show()
    gkdim = L_lbd.GKdim()
    print(gkdim)
    
    