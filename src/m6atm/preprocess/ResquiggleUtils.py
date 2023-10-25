### Modules
import pysam, numba
import pandas as pd, numpy as np, ruptures as rpt


### Config
TraceThres = 5 # adhoc threshold to restrict change point
TraceThresDiff = 1 # adhoc threshold to restrict change point
penalty_value = 100  # beta

SEGMENT_FROM_BASECALLER = 1
SEGMENT_FROM_CHANGE_POINT = 2
SEGMENT_ALL = 3

LOW_THRES_FOR_DEFAULT_TRANS_PROP = 0.6
DIAGONAL_MOVE = 1
HORIZONTAL_MOVE = 2
SKIP_MOVE_BASE = 10
MAXDEL_SIZE = 9
IndelPenalty = 15
IndexExtentionpenalty = 2
BONUS_FOR_EACH_SEGMENT = 1
count = 0

delmargin = 50
margin = 10


### Functions
@numba.jit(nopython=True)
def move_to_index(move, segment_type):

    nl = []
    nl.append(0)
    idx = 1
    for v in move:
        if segment_type == SEGMENT_ALL:
            if v != 0:
                nl.append(idx)
        else: 
            if v == segment_type:
                nl.append(idx)
        idx +=1      
    
    nl = np.array(nl, dtype='int16')
    
    return nl

def move_to_map(move):

    pos_map = {}
    idx = 0
    idx_base = 0
    for v in move:
        if v == SEGMENT_FROM_BASECALLER:
            pos_map[idx_base] = idx
            idx_base +=1
            
        if v > 0:
            idx += 1

    return pos_map

def move_to_tracemap(move):

    pos_map = {}
    idx = 0
    idx_base = 0
    for v in move:
        if v == SEGMENT_FROM_BASECALLER:
            pos_map[idx_base] = idx
            idx_base +=1
            
        idx += 1

    return pos_map

def nearest(pos_list, K):
    # find the element in list with the min of 'abs(list[i] - K)' 
    return pos_list[min(range(len(pos_list)), key = lambda i: abs(pos_list[i] - K))]

def get_near_pos(pos_map, pos):

    if pos in pos_map:
        return pos_map[pos]
    
    keys = list(pos_map.keys())
    nearest_pos = nearest(keys, pos)
    return pos_map[nearest_pos]

def to_compact(trace, potential_mvidx_jit, basecaller_mvidx_jit):
    
    ### trace_compact
    interval_len = len(potential_mvidx_jit)
    trace_len = len(trace)
    trace_compact = np.zeros((interval_len, 5) ,dtype='float32')
    
    idx = 0
    last_basecall_idx = 0
    a, c, g, u = 0, 0, 0, 0

    for m in range(0, trace_len):
        
        trace_compact[idx][4] += 1 # count for length
        if (m>0) and (m in potential_mvidx_jit):
            # update trace_compact
            total = a+c+g+u
            if total>0:
                a = a/total
                c = c/total
                g = g/total
                u = u/total

            trace_compact[idx][0] = a
            trace_compact[idx][1] = c
            trace_compact[idx][2] = g
            trace_compact[idx][3] = u
            idx+=1

            # return to zero
            a, c, g, u = 0, 0, 0, 0

        if (m>0) and (m in basecaller_mvidx_jit):

            num_sep = (idx-1) - last_basecall_idx
            if num_sep>1:
                total_w = 0
                for t in range(last_basecall_idx, idx):
                    total_w += trace_compact[t][4]
                for t in range(last_basecall_idx, idx):
                    trace_compact[t][4] = trace_compact[t][4]/total_w
            else:
                trace_compact[last_basecall_idx][4] = 1.0

            last_basecall_idx = idx

        # accumulate trace_nt value along the trace vector
        trace_nt = trace[m]
        
        _a = max(trace_nt[0], trace_nt[4])
        _c = max(trace_nt[1], trace_nt[5])
        _g = max(trace_nt[2], trace_nt[6])
        _u = max(trace_nt[3], trace_nt[7])

        a = a + _a
        c = c + _c
        g = g + _g
        u = u + _u

    return trace_compact

@numba.jit(nopython=True)
def get_state_prob(ref_jit, trace_compact, n, m):
    
    ref_base = ref_jit[n]
    trace = trace_compact[m]
    
    scA = trace[0]
    scC = trace[1]
    scG = trace[2]
    scU = trace[3]
    weight = trace[4]
    if weight >= 1:
        weight = 1

    total = scA + scC + scG + scU

    if ref_base == 'A': score = scA 
    if ref_base == 'C': score = scC 
    if ref_base == 'G': score = scG 
    if (ref_base == 'T') or (ref_base == 'U'): score = scU
    
    minus_score = 0
    if total > 0:
        plus_score = score/float(total)
        minus_score = (float(total)-score)/float(total)
        
    score = plus_score-minus_score
    score = weight * score

    return score

@numba.jit(nopython=True)
def get_trasition_prob(trace_compact, m):
    trace_before = trace_compact[m-1]
    trace_after = trace_compact[m]
    transition_prob =  (1 - np.dot(trace_before, trace_after)) # dot product as transition probability
    
    # check
    if transition_prob < LOW_THRES_FOR_DEFAULT_TRANS_PROP:
        transition_prob = LOW_THRES_FOR_DEFAULT_TRANS_PROP

    return transition_prob

def get_change_point(trace, move):
       
    # move and trace
    move_t = move.T  
    trace_t = trace.T
    
    # flipflop (A, C, G, T, a, c, g, t)
    for flipflop in trace_t:
        algo_c = rpt.KernelCPD(kernel='linear', min_size=1).fit(flipflop) # trace data: change point detection
        point_list = algo_c.predict(pen = penalty_value)
        
        for idx in point_list:
            if idx>=len(flipflop): break

            # check if order change
            if move_t[idx-1] == 0:
                move_t[idx-1] = SEGMENT_FROM_CHANGE_POINT # annotate change points
    
    cnt = 1
    density0 = len(move_t)//cnt
    last_idx = 0
    for idx in range(len(move_t) - 1):
        if move_t[idx] == SEGMENT_FROM_CHANGE_POINT:
            # clear change point if trace order unchange in small interval
            if not check_change_point(trace, idx) and (idx-last_idx)<density0:
                move_t[idx] = 0

        if move_t[idx] > 0:
            cnt += 1
            
        last_idx = idx
    
    density = len(move_t)//cnt # devide too long segment
    
    idx = 0
    last_idx = 0
    
    for m in move_t:
        if m>0:
            if (idx-last_idx) > density*3:
                unit = (idx-last_idx)//3
                move_t[last_idx+unit] = SEGMENT_FROM_CHANGE_POINT
                move_t[last_idx+(unit*2)] = SEGMENT_FROM_CHANGE_POINT

            elif (idx-last_idx) > density*1.5:
                middle_idx = (idx + last_idx) // 2
                move_t[middle_idx] = SEGMENT_FROM_CHANGE_POINT

            last_idx = idx

        idx += 1

    return move_t


@numba.jit(nopython=True)
def check_change_point(trace, idx):
    # determine if it is change point
    first_idx0, second_idx0 = get_top_two(trace[idx]) # max and second_max
    first_idx1, second_idx1 = get_top_two(trace[idx+1]) # max and second_max
    
    if (first_idx0 == first_idx1) and (second_idx0 == second_idx1):
        return False
    else:
        return True

@numba.jit(nopython=True)
def get_top_two(trace_at_nt):
    # get max and second_max of trace array
    max_val = 0
    max_idx = 0
    second = 0
    second_idx = 0
    idx = -1
    
    for n in trace_at_nt:
        idx+=1
        if n > max_val:
            if second < max_val:
                second = max_val
                second_idx = max_idx
                
            max_val = n
            max_idx = idx
            
        elif n > second:
            second = n
            second_idx = idx
            
    return max_idx, second_idx

@numba.jit(nopython=True)
def range_check(n, m, m_range, banned_interval = 150):

    if n>len(m_range)-1:
        m_in_alignment = m_range[len(m_range)-1]
    else:
        m_in_alignment = m_range[n]

    inrange = (m >= (m_in_alignment-banned_interval)) and (m <= (m_in_alignment + banned_interval))
    diff = abs(m_in_alignment-m)
    wscore = 0
    
    if diff<15:
        wscore = 1 - 0.05*diff

    return inrange, wscore

@numba.jit(nopython=True)
def ViterbiTraceback(mappedref, trace_compact, potential_mvidx, ct_start, m_range):
    # trace_compact
    interval_len = len(trace_compact)
    ref_mapped_len = len(mappedref)

    # score matrix
    score_matrix = np.zeros((ref_mapped_len, interval_len) ,dtype='float32') # ref v.s. trace
    move_matrix = np.zeros((ref_mapped_len, interval_len) ,dtype='float32') # ref v.s. trace
    score_max = 0
    penalty = 0.05

    (max_n, max_m) = (0, 0)

    ### viterbi algorithm: filling the score matrix
    for n in range(ref_mapped_len): 
        for m in range(interval_len):

            inrange, wscore = range_check(n, m, m_range)
            if inrange:
                # transition_score
                if m == 0:
                    score_matrix[n][m] = get_state_prob(mappedref, trace_compact, n, m)

                else:
                    state_prob = get_state_prob(mappedref, trace_compact, n, m)

                    if n==0:
                        untransition_score = score_matrix[n][m-1] + state_prob
                        score_matrix[n][m] = untransition_score
                        move_matrix[n][m] = HORIZONTAL_MOVE

                    untransition_score = score_matrix[n][m-1] + state_prob - penalty
                    transition_score = score_matrix[n-1][m-1] + state_prob

                    # calculate del score
                    del_score_max = 0
                    del_interval_max = 0

                    if n>MAXDEL_SIZE:
                        del_score_max = 0
                        del_interval_max = 0

                        for d_len in range(2, MAXDEL_SIZE+1):
                            del_transition_score = score_matrix[n-d_len][m-1] + state_prob - IndelPenalty - (d_len*IndexExtentionpenalty)
                            if del_transition_score > del_score_max:
                                del_score_max = del_transition_score
                                del_interval_max = d_len

                    # fill score and move
                    if transition_score > untransition_score and transition_score > del_score_max:
                        score_matrix[n][m] = transition_score
                        move_matrix[n][m] = DIAGONAL_MOVE
                    elif untransition_score  > del_score_max:
                        score_matrix[n][m] = untransition_score
                        move_matrix[n][m] = HORIZONTAL_MOVE
                    else:
                        score_matrix[n][m] = del_score_max
                        move_matrix[n][m] = SKIP_MOVE_BASE + del_interval_max # define del_interval_max = 0 at first

                    # score_max record
                    if score_max < score_matrix[n][m]:
                        score_max = score_matrix[n][m]
                        max_n = n
                        max_m = m

            else:
                # skip calculation
                move_matrix[n][m] = DIAGONAL_MOVE


    ### traceback from max 
    left = ref_mapped_len-1-max_n
    traceback = []
    n = max_n
    m = max_m

    countResolvedSegment = 1
    traceback.append((n, potential_mvidx[ct_start+m]))

    while n>=0:
        if move_matrix[n][m]==DIAGONAL_MOVE:
            n = n-1
            m = m-1
            if n>=0 and m>=0:
                traceback.append((n, potential_mvidx[ct_start+m]))
            countResolvedSegment += 1

        elif move_matrix[n][m]>=SKIP_MOVE_BASE:
            d_len = int(move_matrix[n][m] - SKIP_MOVE_BASE)
            m = m-1
            for l in range(1, 1+d_len):
                if n >= 0 and m >= 0:
                    n = n-1
                    traceback.append((n, potential_mvidx[ct_start+m]))
        else:
            # no state change
            m = m-1
            #overwrite
            if m>=0:
                traceback.pop(-1)
                traceback.append((n, potential_mvidx[ct_start+m]))
        if m <= 0:
            break

    ### traceback_reverse
    traceback.reverse()
    
    return traceback

def get_traceboundary(traceback, trace):
    traceboundary = []
    seq = ''
    
    boundary_pre = 0
    idx = 0
    traceboundary.append(boundary_pre)
    for boundary in traceback:
        boundary_val = boundary[1]
        if (boundary_val!=boundary_pre) and (idx>0):
            traceboundary.append(boundary_val)
            seq = seq + get_base_in_boundary(trace, boundary_pre, boundary_val)

        boundary_pre = boundary_val
        idx+=1
        
    seq_len = len(seq)
        
    return traceboundary, seq_len

def get_base_in_boundary(trace, boundary_pre, boundary_val):

    nar = get_trace_in_boundary(trace, boundary_pre, boundary_val)
    nlist = [nar[0], nar[1], nar[2], nar[3]]
    midx = nlist.index(max(nlist))
    base = ['A', 'C', 'G', 'T'][midx]

    return base

def get_trace_in_boundary(trace, boundary_pre, boundary_val):

    acgu = np.zeros(4, dtype='int32')

    for n in range(boundary_pre, boundary_val):

        trace_val = trace[n]
        acgu[0] += trace_val[0]
        acgu[1] += trace_val[1]
        acgu[2] += trace_val[2]
        acgu[3] += trace_val[3]

    return acgu

def correct_cigar(target_pos, cigar):

    a = pysam.AlignedSegment()
    a.cigarstring = cigar
    refpos = 0
    relpos = 0
    
    for cigar_operator, cigar_len in a.cigar:

        if cigar_operator == 3:  # N
            refpos = refpos + cigar_len

        elif cigar_operator == 0 or cigar_operator == 4:  # match or S softclip was not correted so treat as M
            if refpos + cigar_len > target_pos:
                return relpos + (target_pos - refpos)

            relpos = relpos + cigar_len
            refpos = refpos + cigar_len

        elif cigar_operator == 2:  # Del
            refpos = refpos + cigar_len

        elif cigar_operator == 1:  # Ins
            if relpos == 0:
                if target_pos <= cigar_len:
                    return 0

            relpos = relpos + cigar_len

    return 0

def to_cigar(traceback, seq_len):

    n_prev = -1
    m_prev = -1
    cnt_total = 0
    cigar_list = []
    count_del = 0
    count_match_mismatch = 0
    init = True
    
    for n, m in traceback:
        if init:
            if n>0:
                cigar_list.append(str(n) + 'N')
            init = False

        if m == m_prev:  # deletion
            count_del+=1
            if (count_match_mismatch>0):
                if (cnt_total+count_match_mismatch)>seq_len:
                    mm = seq_len-cnt_total
                else:
                    mm = count_match_mismatch
                cnt_total += mm
                cigar_list.append(str(mm) + 'M')

            count_match_mismatch = 0

        else:  # match or mismatch
            count_match_mismatch += 1
            if (count_del > 0):
                cigar_list.append(str(count_del) + 'D')
            count_del = 0
        m_prev = m

    if (count_match_mismatch > 0):
        if (cnt_total + count_match_mismatch) > seq_len:
            mm = seq_len - cnt_total
        else:
            mm = count_match_mismatch
        cnt_total += mm
        cigar_list.append(str(mm) + 'M')
    if (count_del > 0):
        cigar_list.append(str(count_del) + 'D')
        
    cigar_string = ''.join(cigar_list)

    return cigar_string

