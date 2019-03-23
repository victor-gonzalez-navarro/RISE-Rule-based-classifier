import numpy as np


# -------------------------------------------------------------------------------------------------- AUXILIARY FUNCTIONS
# Compute similarity between different values of an attributes
def computeSVDM(x, y, data, labels, set_labels, N_clases, atribute_num):
    res = 0
    for j in range(N_clases):
        clas = set_labels[j]
        Nax = np.sum(data[:,atribute_num] == x)
        Nay = np.sum(data[:,atribute_num] == y)
        Naxc = np.sum((data[:,atribute_num] == x)*(labels==clas))
        Nayc = np.sum((data[:,atribute_num] == y)*(labels==clas))
        new_value = abs((Naxc/Nax)-(Nayc/Nay))
        res = res + (new_value)
    return res


# Compute the distance between 1 rule and 1 instance
def distance_R_I(rule, inst, dist_measure, rule2, numoricalatt):
    distance = 0
    for k in range(len(inst)-1):
        if (numoricalatt[k] == 0):
            if not (rule[k] == -1):
                distance = distance + dist_measure[k][str(rule[k])+'-'+str(inst[k])]
        else:
            if (inst[k] > rule2[k]):
                distance = abs(inst[k] - rule2[k])
            elif (inst[k] < rule[k]):
                distance = abs(inst[k] - rule[k])
            elif (inst[k] >= rule[k] and (inst[k] <= rule2[k])):
                distance = 0

    return distance


# Useful to delete duplicated rules
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


# Find the closest rule for each instance so as to compute the Accuracy
def compute_accuracy(ES, RS, dist_measure, principi, N_clases, RS2, numoricalatt):
    idx_inst_to_rule = []
    inst_to_rule = []
    inst_to_rule2 = []
    N_instances = ES.shape[0]
    # For each instance
    for i in range(N_instances):
        dist_vector = []
        # For each rule
        for j in range(RS.shape[0]):
            # Accuracy is measured using a leave-one-out methodology: when attempting to classify an example,
            # the corresponding rule is left out, unless it already covers other examples as well.
            distance = 258

            # ************** Useful to avoid problems with datasets that mix symbolic and numrical attributes
            eequal = True
            for kat in range(ES.shape[1] - 1):
                if (numoricalatt[kat]==0) and (RS[j,kat] != ES[i,kat]):
                    eequal = False
                if (numoricalatt[kat]==1) and ((RS[j,kat] != ES[i,kat])or(RS2[j,kat] != ES[i,kat])):
                    eequal = False
            # **************

            if not (eequal):
                distance = 0
                # For each attribute
                for k in range(ES.shape[1] - 1):
                    if (numoricalatt[k] == 0):
                        if (RS[j, k] != -1):
                            distance = distance + dist_measure[k][str(RS[j, k]) + '-' + str(ES[i, k])]
                    else:
                        if (ES[i, k] > RS2[j, k]):
                            distance = abs(ES[i, k]-RS2[j, k])
                        elif (ES[i, k] < RS[j, k]):
                            distance = abs(ES[i, k]-RS[j, k])
                        elif (ES[i, k] >= RS[j, k] and (ES[i, k] <= RS2[j, k])):
                            distance = 0


            dist_vector = dist_vector + [distance,]

        # Apply Laplace correction when there is a tie (in all iterations except the first one)
        if principi:
            idx_min = np.argmin(np.array(dist_vector))
        else:
            # Is more than one rule is at the same distance from the instance?
            indices_min = np.where(dist_vector == np.amin(dist_vector))[0]
            if len(indices_min) == 1:
                idx_min = indices_min[0]
            else:
                # Choose the more general rule but that classifies correctly (Laplace correction)
                # idx_min = indices_min[0]
                idx_min = laplace_correction(ES, RS, indices_min, dist_measure, N_clases, RS2, numoricalatt)

        idx_inst_to_rule = idx_inst_to_rule + [idx_min, ]
        inst_to_rule = inst_to_rule + [RS[idx_min, :], ]
        inst_to_rule2 = inst_to_rule2 + [RS2[idx_min, :], ]

    # Compute accuracy using inst_to_rule
    accuracy = 0
    for i in range(N_instances):
        if ES[i, -1:] == RS[idx_inst_to_rule[i], -1:]:
            accuracy = accuracy + 1
    precision_final = accuracy / N_instances
    return precision_final, inst_to_rule, inst_to_rule2


# Laplace correction: find the best rule (covers more and is accurate) when several are equally near
def laplace_correction(ES, RS, indices_min, dist_measure, N_clases, RS2, numoricalatt):
    num_inst_covered = []
    N_instances = ES.shape[0]
    for itt in range(len(indices_min)):
        nplus = 0
        nminus = 0
        for ii in range(N_instances):
            distance = 0
            # For each attribute
            for kk in range(ES.shape[1] - 1):
                if (numoricalatt[kk] == 0):
                    if (RS[indices_min[itt], kk] != -1):
                        distance = distance + dist_measure[kk][str(RS[indices_min[itt], kk]) + '-' + str(ES[ii, kk])]
                else:
                    if (ES[ii, kk] > RS2[indices_min[itt], kk]):
                        distance = abs(ES[ii, kk] - RS2[indices_min[itt], kk])
                    elif (ES[ii, kk] < RS[indices_min[itt], kk]):
                        distance = abs(ES[ii, kk] - RS[indices_min[itt], kk])
                    elif (ES[ii, kk] >= RS[indices_min[itt], kk] and (ES[ii, kk] <= RS2[indices_min[itt], kk])):
                        distance = 0

            # The rule covers this instance
            if distance == 0:
                # The instance that the rule covers has the same class as the class of the rule
                if (ES[ii, -1] == RS[indices_min[itt], -1]):
                    nplus = nplus + 1
                else:
                    nminus = nminus + 1
        # Compute H value for specific rule to know if it general but also accurate
        H = (nplus + 1)/(nplus + nminus + N_clases)
        num_inst_covered = num_inst_covered + [H,]
    best_rule = np.argmax(np.array(num_inst_covered))
    return indices_min[best_rule]


# Useful to display the rules in the screen
def print_ruleSet(RS, num_inst_covered, precision_byRule, RS2, numoricalatt):
    print('')
    for ruleid in range(RS.shape[0]):
        spa = ' '*(len(str(RS.shape[0]))+1-len(str(ruleid)))
        print('Rule'+spa + str(ruleid) + ':'+'\033[1m'+'[IF] ', end='')

        for att in range(RS.shape[1]-1):
            if (numoricalatt[att] == 0):
                if RS[ruleid,att] == -1:
                    print('\033[1m'+'At' + str(att) + '\033[0m'+'=True', end = "      ")
                else:
                    print('\033[1m'+'At' + str(att) + '\033[0m'+ '='+str(RS[ruleid,att])+'      ', end = " ")
            else:
                spaces = ' '*(12-3-len(str(round(RS[ruleid, att],2)))-len(str(round(RS2[ruleid, att],2))))
                print('\033[1m' + 'At' + str(att) + '\033[0m' + '=[' + str(round(RS[ruleid, att],2))+','+str(round(
                    RS2[ruleid, att],2)) + ']', end=spaces)

        spa2 = ' '*(len(str(RS.shape[0]))+1-len(str(num_inst_covered[ruleid])))
        print('\033[1m'+' [THEN] Class: '+'\033[0m'+str(RS[ruleid,-1])+'\033[1m'+'     *** Coverage: '+'\033[0m'+str(
            num_inst_covered[ruleid])+'\033[1m'+spa2+'* Precision: '+'\033[0m'+str(round(precision_byRule[ruleid],3)))


def classify_tst(ES, RS, dist_measure, RS2, numoricalatt):
    idx_inst_to_rule = []
    N_instances = ES.shape[0]
    labb = []
    # For each instance
    for i in range(N_instances):
        dist_vector = []
        # For each rule
        for j in range(RS.shape[0]):
            distance = 0
            # For each attribute
            for k in range(ES.shape[1]):
                if (numoricalatt[k] == 0):
                    if (RS[j, k] != -1):
                        distance = distance + dist_measure[k][str(RS[j, k]) + '-' + str(ES[i, k])]
                else:
                    if (ES[i, k] > RS2[j, k]):
                        distance = abs(ES[i, k] - RS2[j, k])
                    elif (ES[i, k] < RS[j, k]):
                        distance = abs(ES[i, k] - RS[j, k])
                    elif (ES[i, k] >= RS[j, k] and (ES[i, k] <= RS2[j, k])):
                        distance = 0

            dist_vector = dist_vector + [distance, ]

        # Is more than one rule is at the same distance from the instance?
        indices_min = np.where(dist_vector == np.amin(dist_vector))[0]
        counts = np.bincount(indices_min)
        idx_min = np.argmax(counts) # Chose the most common label

        idx_inst_to_rule = idx_inst_to_rule + [idx_min, ]
        labb = labb + [RS[idx_inst_to_rule[i], -1:][0], ]

    return np.array(labb)


# Compute coverage and precision
def compute_coverage_precision(ES, RS, RS2, numoricalatt, dist_measure):
    num_inst_covered = []
    precision_byRule = []
    N_instances = ES.shape[0]
    N_rules = RS.shape[0]
    for rule in range(N_rules):
        nplus = 0
        nminus = 0
        ints_cover = 0
        for inst in range(N_instances):
            distance = 0
            # For each attribute
            for kk in range(ES.shape[1] - 1):
                if (numoricalatt[kk] == 0):
                    if (RS[rule, kk] != -1):
                        distance = distance + dist_measure[kk][str(RS[rule, kk]) + '-' + str(ES[inst, kk])]
                else:
                    if (ES[inst, kk] > RS2[rule, kk]):
                        distance = abs(ES[inst, kk] - RS2[rule, kk])
                    elif (ES[inst, kk] < RS[rule, kk]):
                        distance = abs(ES[inst, kk] - RS[rule, kk])
                    elif (ES[inst, kk] >= RS[rule, kk] and (ES[inst, kk] <= RS2[rule, kk])):
                        distance = 0

            # The rule covers this instance
            if distance == 0:
                ints_cover = ints_cover + 1
                # The instance that the rule covers has the same class as the class of the rule
                if (ES[inst, -1] == RS[rule, -1]):
                    nplus = nplus + 1
                else:
                    nminus = nminus + 1
        # Compute coverage and precision per rule
        num_inst_covered = num_inst_covered + [ints_cover, ]
        precision_byRule = precision_byRule + [(nplus/(nplus+nminus)), ]
    return num_inst_covered, precision_byRule
