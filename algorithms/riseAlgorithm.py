from algorithms.auxiliaryFunctions import *

class riseAlgorithm():

    trn_data_RS = None
    trn_data_RS2 = None
    trn_data_ES = None
    distmeasure = None
    tst_labels = None

    def __init__(self, numoricalatt, cut):
        self.numoricalatt = numoricalatt
        self.cut = cut

    def fit(self, data, labels):

        # Save in a dictionary the distance between the nominal values for each atribute [SVDM distance]
        N_instances = data.shape[0]
        d_atributes = data.shape[1]
        N_clases = len(set(labels))
        set_labels = list(set(labels))

        dist_measure = dict()
        for i in range(d_atributes):
            # In case the attribute is symbolic we need to compute SVDM
            if self.numoricalatt[i] == 0:
                listval = list(set(data[:, i]))
                dist_subatribute = dict()
                for p1 in range(len(listval)):
                    for p2 in range(p1, len(listval)):
                        if listval[p1] == listval[p2]:
                            dist_subatribute[str(listval[p1]) + '-' + str(listval[p2])] = 0
                        else:
                            res = computeSVDM(listval[p1], listval[p2], data, labels, set_labels, N_clases, i)
                            dist_subatribute[str(listval[p1]) + '-' + str(listval[p2])] = res
                            dist_subatribute[str(listval[p2]) + '-' + str(listval[p1])] = res
                dist_measure[i] = dist_subatribute
            # The attribute is numerical and therefore we do not have to compute the SVDM distance
            else:
                dist_measure[i] = 'Numerical'

        # Beginning of the train algorithm
        ES = np.concatenate((data, labels.reshape(N_instances, 1)), axis=1)

        # Let RS be ES
        RS = ES
        RS2 = ES
        # Compute Acc(RS)
        (precision_final, inst_to_rule, inst_to_rule2) = compute_accuracy(ES, RS, dist_measure, True, N_clases, RS2,
                                                                          self.numoricalatt)
        print('The initial Train accuracy is: ' + str(round(precision_final, 3)))
        print('------------IERATION 1------------')

        # While Precision_final ≤ Precision_initial
        precision_init = -1
        counter_number = 1
        while (precision_final > precision_init):
            RSanterior = RS
            RSanterior2 = RS2

            precision_init = precision_final

            # For each rule:
            for rule in range(RS.shape[0]):
                # Find nearest instance to the rule R not covered by the rule and being of the same class
                minimum = 10
                idx_min = -10
                for inst in range(ES.shape[0]):
                    distance = distance_R_I(RS[rule, :], ES[inst, :], dist_measure, RS2[rule, :], self.numoricalatt)
                    # Not covered by the rule = (distance > 0) & Being of the same class = (RS[rule,-1:]==ES[inst,-1:])
                    if (distance < minimum) and (distance > 0) and (RS[rule, -1:] == ES[inst, -1:]):
                        minimum = distance
                        idx_min = inst
                nearest_inst = idx_min

                # Generalize the rule given the nearest instance
                R_generalized = np.copy(RS[rule, :])
                R_generalized2 = np.copy(RS2[rule, :])
                for k in range(d_atributes):
                    if (self.numoricalatt[k] == 0):
                        if (RS[rule, k] != -1) and (RS[rule, k] != ES[nearest_inst, k]):
                            R_generalized[k] = -1
                    else:
                        if (ES[nearest_inst, k] > RS2[rule, k]):
                            R_generalized2[k] = ES[nearest_inst, k]
                        elif (ES[nearest_inst, k] < RS[rule, k]):
                            R_generalized[k] = ES[nearest_inst, k]

                RSprima = np.copy(RS)
                RSprima[rule, :] = np.copy(R_generalized)

                RSprima2 = np.copy(RS2)
                RSprima2[rule, :] = np.copy(R_generalized2)

                # IF Acc(RS') >= Acc(RS)
                increment_accuracy = 0
                for inst in range(ES.shape[0]):
                    # Does the new rule wins a new instance that it did not before? (win) and (before was different)
                    precision_RSprima = distance_R_I(R_generalized, ES[inst, :], dist_measure, R_generalized2,
                                                     self.numoricalatt)
                    precision_RS = distance_R_I(inst_to_rule[inst], ES[inst, :], dist_measure, inst_to_rule2[inst],
                                                self.numoricalatt)

                    if (precision_RSprima < precision_RS) and (
                    not (np.array_equal(inst_to_rule[inst], R_generalized))) and (
                            not (np.array_equal(inst_to_rule2[inst], R_generalized2))):
                        # A previously missclasified example is now correctly classified
                        if (R_generalized[-1] == ES[inst, -1]) and (inst_to_rule[inst][-1] != ES[inst, -1]):
                            increment_accuracy = increment_accuracy + 1
                        # A previously correctly classified example is now missclisified
                        elif (R_generalized[-1] != ES[inst, -1]) and (inst_to_rule[inst][-1] == ES[inst, -1]):
                            increment_accuracy = increment_accuracy - 1
                        # As the paper says: "the relevant structures are updated"
                        inst_to_rule[inst] = np.copy(R_generalized)
                        inst_to_rule2[inst] = np.copy(R_generalized2)

                # If Acc(RS') >= Acc(RS) --> Replace RS by RS'
                if increment_accuracy >= 0:
                    RS = np.copy(RSprima)
                    RS2 = np.copy(RSprima2)

            # Delete repeated rules. This statement needs to be here to avoid modifying RS while generalizing some rules
            numoricalattsec = np.array(list(self.numoricalatt) + [0, ])
            borrar_rows = []
            iterr1 = 0
            while iterr1 < (RS.shape[0]):
                equalr = True
                iterr2 = 0
                finiquito = False
                while ((iterr2 < (RS.shape[0])) and (finiquito == False)):
                    for atrr in range(d_atributes + 1):
                        if (numoricalattsec[atrr] == 0):
                            if (RS[iterr1, atrr] != RS[iterr2, atrr]):
                                equalr = False
                        else:
                            if (RS[iterr1, atrr] != RS[iterr2, atrr]) or (RS2[iterr1, atrr] != RS2[iterr2, atrr]):
                                equalr = False
                    if equalr:
                        borrar_rows = borrar_rows + [iterr1, ]
                        finiquito = True
                    iterr2 = iterr2 + 1
                iterr1 = iterr1 + 1

            RS = np.delete(RS, borrar_rows, axis=0)
            RS2 = np.delete(RS2, borrar_rows, axis=0)

            # Precision_final = Acc(RS)
            (precision_final, inst_to_rule, inst_to_rule2) = compute_accuracy(ES, RS, dist_measure, False, N_clases,
                                                                              RS2, self.numoricalatt)
            print('The Train accuracy is: ' + str(round(precision_final, 3)))
            counter_number = counter_number + 1
            print('------------IERATION ' + str(counter_number) + '------------')

        self.trn_data_RS = RSanterior
        self.trn_data_RS2 = RSanterior2
        self.distmeasure = dist_measure
        self.trn_data_ES = ES

    def classify(self, data_test):
        tst_labels = classify_tst(data_test, self.trn_data_RS, self.distmeasure,
                                              self.trn_data_RS2, self.numoricalatt)
        self.tst_labels = tst_labels

    def print_rules(self):
        num_inst_covered, precision_byRule = compute_coverage_precision(self.trn_data_ES, self.trn_data_RS,
                                                self.trn_data_RS2, self.numoricalatt, self.distmeasure)
        print_ruleSet(self.trn_data_RS, num_inst_covered, precision_byRule, self.trn_data_RS2, self.numoricalatt)