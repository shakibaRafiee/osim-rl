# - [x y z] -> [anterior lateral superior]
#         (<-> [posterior medial inferior])

from __future__ import division # '/' always means non-truncating division
import numpy as np

class LocoCtrl(object):
    DEBUG = 0

    RIGHT = 0 # r_leg
    LEFT = 1 # l_leg

    # (todo) use these when handling angles
    # THETA0 = 0*np.pi/180 # trunk angle when standing straight
    # S_THETA = 1 # 1: leaning forward > 0; -1: leaning backward > 0
    # HIP0 = 0*np.pi/180 # hip angle when standing straight
    # S_HIP = 1 # 1: extension > 0; -1: flexion > 0
    # KNEE0 = 0*np.pi/180 # knee angle when standing straight
    # S_KNEE = 1 # 1: extension > 0; -1: flexion > 0
    # ANKLE0 = 0*np.pi/180 # ankle angle when standing straight
    # S_ANKLE = 1 # 1: plantar flexion > 0; -1: dorsiflexion > 0
    
    # muscle names
    m_keys = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']
    
    # body sensor data
    s_b_keys = ['theta', 'd_pos', 'dtheta']
        # theta[0]: around local x axis (pointing anterior)
        # theta[1]: around local y axis (pointing leftward)
        # theta[2]: around local z axis (pointing upward)
        # pos[0]: local x
        # pos[1]: local y
        # pos[2]: local z
        
    # leg sensor data
    # anglular values follow the Song2015 convention
    s_l_keys = [
        'contact_ipsi', 'contact_contra', 'load_ipsi', 'load_contra',
        'alpha', 'alpha_f', 'dalpha',
        'phi_hip', 'phi_knee', 'phi_ankle', 'dphi_knee'
        'F_RF', 'F_VAS', 'F_GAS', 'F_SOL',
        ]
    
    # control states
    cs_keys = [
        'ph_st', # leg in stance
        'ph_st_csw', # leg in stance ^ contra-leg in swing
        'ph_st_sw0', # leg in stance ^ initial swing
        'ph_sw', # leg in swing
        'ph_sw_flex_k', # leg in swing ^ flex knees
        'ph_sw_hold_k', # leg in swing ^ hold knee
        'ph_sw_stop_l', # leg in swing ^ stop leg
        'ph_sw_hold_l' # leg in swing ^ hold leg
        ]
    
    # control parameters
    cp_keys = [
        'theta_tgt', 'c0', 'cv', 'alpha_delta',
        'knee_sw_tgt', 'knee_tgt', 'knee_off_st', 'ankle_tgt',
        'HFL_3_PG', 'HFL_3_DG', 'HFL_6_PG', 'HFL_6_DG', 'HFL_10_PG',
        'GLU_3_PG', 'GLU_3_DG', 'GLU_6_PG', 'GLU_6_DG', 'GLU_10_PG',
        'HAM_3_GLU', 'HAM_9_PG',
        'RF_1_FG', 'RF_8_DG_knee',
        'VAS_1_FG', 'VAS_2_PG', 'VAS_10_PG',
        'BFSH_2_PG', 'BFSH_7_DG_alpha', 'BFSH_7_PG', 'BFSH_8_DG', 'BFSH_8_PG',
        'BFSH_9_G_HAM', 'BFSH_9_HAM0', 'BFSH_10_PG',
        'GAS_2_FG',
        'SOL_1_FG',
        'TA_5_PG', 'TA_5_G_SOL',
        'theta_tgt_f', 'c0_f', 'cv_f',
        'HAB_3_PG', 'HAB_3_DG', 'HAB_6_PG',
        'HAD_3_PG', 'HAD_3_DG', 'HAD_6_PG'
        ]
    
    par_space = (
        [0.0, -1.0, 0.0, 0.0, \
        -2.0, -90/15, -1.0, 0.0, \
        0.0, 0.0, 0.0, 0.0, 0.0, \
        0.0, 0.0, 0.0, 0.0, 0.0, \
        0.0, 0.0, \
        0.0, 0.0, \
        0.0, 0.0, 0.0, \
        0.0, 0.0, 0.0, 0.0, 0.0, \
        0.0, 0.0, 0.0, \
        0.0, \
        0.0, \
        0.0, 0.0, \
        0.0, 0.0, \
        0.0, 0.0, 0.0, \
        0.0, 0.0, 0.0],
        [6.0, 3.0, 5.0, 3.0, \
        3.0, 20/15, 15/10, 3.0, \
        3.0, 3.0, 3.0, 3.0, 3.0, \
        3.0, 3.0, 3.0, 3.0, 3.0, \
        3.0, 3.0, \
        3.0, 3.0, \
        3.0, 3.0, 3.0, \
        3.0, 3.0, 3.0, 3.0, 3.0, \
        3.0, 3.0, 3.0, \
        3.0, \
        3.0, \
        3.0, 3.0, \
        2.0, 3.0, \
        3.0, 3.0, 3.0, \
        3.0, 3.0, 3.0])
    
    m_map = dict(zip(m_keys, range(len(m_keys))))
    s_b_map = dict(zip(s_b_keys, range(len(s_b_keys))))
    s_l_map = dict(zip(s_l_keys, range(len(s_l_keys))))
    cs_map = dict(zip(cs_keys, range(len(cs_keys))))
    cp_map = dict(zip(cp_keys, range(len(cp_keys))))
    
    #------------------------------------------------------------------------------------------------
    # to make an object
    def __init__(self, TIMESTEP, control_mode=1, control_dimension=3, params=np.ones(len(cp_keys)), W_CPG= 0, phi_CPG= 0):
        if self.DEBUG:
            print("===========================================")
            print("locomotion controller created in DEBUG mode")
            print("===========================================")

        self.control_mode = control_mode
        # 0: spinal control (no brain control)
        # 1: full control
        self.control_dimension = control_dimension # 2D or 3D
        
        #Shakiba
        self.W_CPG = W_CPG
        self.phi_CPG = phi_CPG    # d_phi on/off for each rectangular wave
        #
        if self.control_mode == 0:
            self.brain_control_on = 0
        elif self.control_mode == 1:
            self.brain_control_on = 1

        self.spinal_control_phase = {}
        self.in_contact = {}
        self.brain_command = {}
        self.stim = {}
        
        #Shakiba
        self.stim0_ph = {}
        self.Phase = {}
        self.TIMESTEP = TIMESTEP
        
        self.omega = (1/1.05)*(2*np.pi)              # Song model walking period is 1.05, we originally used 1, but changed it later(11/11/2020)
                
        self.cutan = {}
        self.cutan['t_Con'] = {}
        self.cutan['t_Off'] = {}
        #

        self.n_par = len(LocoCtrl.cp_keys)
        if self.control_dimension == 2:
            self.n_par = 37
            self.par_space = (self.par_space[0][0:37], self.par_space[1][0:37])
        self.cp = {}

        self.reset(params)
    #------------------------------------------------------------------------------------------------
    # reset specifies dicts and spinal_control_phase and stim
    def reset(self, params=None):
        self.in_contact['r_leg'] = 1
        self.in_contact['l_leg'] = 0

        spinal_control_phase_r = {}
        spinal_control_phase_r['ph_st'] = 1
        spinal_control_phase_r['ph_st_csw'] = 0
        spinal_control_phase_r['ph_st_sw0'] = 0
        spinal_control_phase_r['ph_st_st'] = 0
        spinal_control_phase_r['ph_sw'] = 0
        spinal_control_phase_r['ph_sw_flex_k'] = 0
        spinal_control_phase_r['ph_sw_hold_k'] = 0
        spinal_control_phase_r['ph_sw_stop_l'] = 0
        spinal_control_phase_r['ph_sw_hold_l'] = 0
        self.spinal_control_phase['r_leg'] = spinal_control_phase_r

        spinal_control_phase_l = {}
        spinal_control_phase_l['ph_st'] = 0
        spinal_control_phase_l['ph_st_csw'] = 0
        spinal_control_phase_l['ph_st_sw0'] = 0
        spinal_control_phase_l['ph_st_st'] = 0
        spinal_control_phase_l['ph_sw'] = 1
        spinal_control_phase_l['ph_sw_flex_k'] = 1
        spinal_control_phase_l['ph_sw_hold_k'] = 0
        spinal_control_phase_l['ph_sw_stop_l'] = 0
        spinal_control_phase_l['ph_sw_hold_l'] = 0
        self.spinal_control_phase['l_leg'] = spinal_control_phase_l

        self.stim['r_leg'] = dict(zip(self.m_keys, 0.01*np.ones(len(self.m_keys))))
        self.stim['l_leg'] = dict(zip(self.m_keys, 0.01*np.ones(len(self.m_keys))))

        if params is not None:
            self.set_control_params(params)
    #------------------------------------------------------------------------------------------------
    # based on n_par specifies the control parameters
    def set_control_params(self, params):
        if len(params) == self.n_par:
            self.set_control_params_RL('r_leg', params)
            self.set_control_params_RL('l_leg', params)
        elif len(params) == 2*self.n_par:
            self.set_control_params_RL('r_leg', params[:self.n_par])
            self.set_control_params_RL('l_leg', params[self.n_par:])
        else:
            raise Exception('error in the number of params!!')
    # -----------------------------------------------------------------------------------------------------------------
    # specifies the control parameters (we probably want to add to this list, I don't know how yet)
    def set_control_params_RL(self, s_leg, params):
        cp = {}
        cp_map = self.cp_map

        cp['theta_tgt'] = params[cp_map['theta_tgt']] *10*np.pi/180
        cp['c0'] = params[cp_map['c0']] *20*np.pi/180 +55*np.pi/180
        cp['cv'] = params[cp_map['cv']] *2*np.pi/180
        cp['alpha_delta'] = params[cp_map['alpha_delta']] *5*np.pi/180
        cp['knee_sw_tgt'] = params[cp_map['knee_sw_tgt']] *20*np.pi/180 +120*np.pi/180
        cp['knee_tgt'] = params[cp_map['knee_tgt']] *15*np.pi/180 +160*np.pi/180
        cp['knee_off_st'] = params[cp_map['knee_off_st']] *10*np.pi/180 +165*np.pi/180
        cp['ankle_tgt'] = params[cp_map['ankle_tgt']] *20*np.pi/180 +60*np.pi/180

        cp['HFL_3_PG'] = params[cp_map['HFL_3_PG']] *2.0
        cp['HFL_3_DG'] = params[cp_map['HFL_3_DG']] *1.0
        cp['HFL_6_PG'] = params[cp_map['HFL_6_PG']] *1.0
        cp['HFL_6_DG'] = params[cp_map['HFL_6_DG']] *.1
        cp['HFL_10_PG'] = params[cp_map['HFL_10_PG']] *1.0

        cp['GLU_3_PG'] = params[cp_map['GLU_3_PG']] *2.0
        cp['GLU_3_DG'] = params[cp_map['GLU_3_DG']] *0.5
        cp['GLU_6_PG'] = params[cp_map['GLU_6_PG']] *1.0
        cp['GLU_6_DG'] = params[cp_map['GLU_6_DG']] *0.1
        cp['GLU_10_PG'] = params[cp_map['GLU_10_PG']] *.5

        cp['HAM_3_GLU'] = params[cp_map['HAM_3_GLU']] *1.0
        cp['HAM_9_PG'] = params[cp_map['HAM_9_PG']] *2.0

        cp['RF_1_FG'] = params[cp_map['RF_1_FG']] *0.3
        cp['RF_8_DG_knee'] = params[cp_map['RF_8_DG_knee']] *0.1

        cp['VAS_1_FG'] = params[cp_map['VAS_1_FG']] *1.0
        cp['VAS_2_PG'] = params[cp_map['VAS_2_PG']] *2.0
        cp['VAS_10_PG'] = params[cp_map['VAS_10_PG']] *.3

        cp['BFSH_2_PG'] = params[cp_map['BFSH_2_PG']] *2.0
        cp['BFSH_7_DG_alpha'] = params[cp_map['BFSH_7_DG_alpha']] *0.2
        cp['BFSH_7_PG'] = params[cp_map['BFSH_7_PG']] *2.0
        cp['BFSH_8_DG'] = params[cp_map['BFSH_8_DG']] *1.0
        cp['BFSH_8_PG'] = params[cp_map['BFSH_8_DG']] *1.0
        cp['BFSH_9_G_HAM'] = params[cp_map['BFSH_9_G_HAM']] *2.0
        cp['BFSH_9_HAM0'] = params[cp_map['BFSH_9_HAM0']] *0.3
        cp['BFSH_10_PG'] = params[cp_map['BFSH_10_PG']] *2.0

        cp['GAS_2_FG'] = params[cp_map['GAS_2_FG']] *1.2

        cp['SOL_1_FG'] = params[cp_map['SOL_1_FG']] *1.2

        cp['TA_5_PG'] = params[cp_map['TA_5_PG']] *2.0
        cp['TA_5_G_SOL'] = params[cp_map['TA_5_G_SOL']] *0.5

        if self.control_dimension == 3:
            if len(params) != 46:
                raise Exception('error in the number of params!!')
            cp['theta_tgt_f'] = params[cp_map['theta_tgt_f']] *5.0*np.pi/180
            cp['c0_f'] = params[cp_map['c0_f']] *20*np.pi/180 + 60*np.pi/180
            cp['cv_f'] = params[cp_map['cv_f']] *10*np.pi/180
            cp['HAB_3_PG'] = params[cp_map['HAB_3_PG']] *10.0
            cp['HAB_3_DG'] = params[cp_map['HAB_3_DG']] *1
            cp['HAB_6_PG'] = params[cp_map['HAB_6_PG']] *2.0
            cp['HAD_3_PG'] = params[cp_map['HAD_3_PG']] *2.0
            cp['HAD_3_DG'] = params[cp_map['HAD_3_DG']] *0.3
            cp['HAD_6_PG'] = params[cp_map['HAD_6_PG']] *2.0
        elif self.control_dimension == 2:
            if len(params) != 37:
                raise Exception('error in the number of params!!')

        self.cp[s_leg] = cp

# -----------------------------------------------------------------------------------------------------------------
    def update(self, sensor_data, t_c, Phase, cutan):
        self.sensor_data = sensor_data
        
        if self.brain_control_on:
            # update self.brain_command
            self._brain_control(sensor_data, t_c, Phase, cutan)   # Shakiba: added Phase as an input for brain_control
        
        # updates self.stim
        self._spinal_control(sensor_data)

        stim = np.array([self.stim['r_leg']['HFL'], self.stim['r_leg']['GLU'],
            self.stim['r_leg']['HAM'], self.stim['r_leg']['RF'],
            self.stim['r_leg']['VAS'], self.stim['r_leg']['BFSH'],
            self.stim['r_leg']['GAS'], self.stim['r_leg']['SOL'],
            self.stim['r_leg']['TA'],
            self.stim['l_leg']['HFL'], self.stim['l_leg']['GLU'],
            self.stim['l_leg']['HAM'], self.stim['l_leg']['RF'],
            self.stim['l_leg']['VAS'], self.stim['l_leg']['BFSH'],
            self.stim['l_leg']['GAS'], self.stim['l_leg']['SOL'],
            self.stim['l_leg']['TA']
            ])
        # todo: self._flaten(self.stim)
        
        # Shakiba
        Phase = self.Phase
        cutan = self.cutan
        #
        
        return stim, Phase, cutan    # Shakiba: added Phase as both the input and output of update

# -----------------------------------------------------------------------------------------------------------------
    # brain_control uses sensory data(without delay) to specify the brain_commands
    
    # (Song & Geyer 2015)
    # The supraspinal layer adjusts the desired foot placements (alpha_tgt) and the desired minimum swing
    # leg length (lclr), and selects which leg should transition into swing control during double support.
    # B, desired foot placement is calculated as target leg angles αtgt s and αtgt f for sagittal (s) and frontal (f)
    # plane motions based on the velocity vs,f of the COM and its distance to the stance leg ankle, ds,f.
    
    def _brain_control(self, sensor_data=0, t_c = 0, Phase = 0, cutan = 0):
        s_b = sensor_data['body']
        cp = self.cp
        
        self.brain_command['r_leg'] = {}
        self.brain_command['l_leg'] = {}
        
        #Shakiba
        self.Phase['r_leg'] = Phase['r_leg']
        self.Phase['l_leg'] = Phase['l_leg']
        
        self.cutan['t_Con']['r_leg']= cutan['t_Con']['r_leg']
        self.cutan['t_Off']['l_leg']= cutan['t_Off']['l_leg']
        self.cutan['t_Con']['l_leg']= cutan['t_Con']['l_leg']
        self.cutan['t_Off']['r_leg']= cutan['t_Off']['r_leg']
        
        self.stim0_ph['r_leg'] = {}
        self.stim0_ph['l_leg'] = {}
                
        muscle = ['HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']
        
        phi_CPG = self.phi_CPG
        #(I was multiplying phi by omega, which made sanse for the implementation before dirac, but not after, this was corrected at 6/4/2021 after W22)
        #(unfortunately, even though it does not make sense, I had to revert it back, for the model converge to a reasonable solution)
        '''
        phi  = np.array([phi_CPG[0], np.sum(phi_CPG[0:3]), np.sum(phi_CPG[0:5]), np.sum(phi_CPG[0:7]), np.sum(phi_CPG[0:9])])*self.omega
        dphi = np.array([phi_CPG[1], phi_CPG[3], phi_CPG[5], phi_CPG[7], phi_CPG[9]])*self.omega

        self.phi_Con = 0.0493*self.omega
        self.phi_Off = 0.5650*self.omega
        '''
        phi  = np.array([phi_CPG[0], np.sum(phi_CPG[0:3]), np.sum(phi_CPG[0:5]), np.sum(phi_CPG[0:7]), np.sum(phi_CPG[0:9])])*(2*np.pi) 
        dphi = np.array([phi_CPG[1], phi_CPG[3], phi_CPG[5], phi_CPG[7], phi_CPG[9]])*(2*np.pi) 

        self.phi_Con = 0.0493*(2*np.pi) 
        self.phi_Off = 0.5650*(2*np.pi) 
        #'''
        w = {}
        w = self.W_CPG
        #
        
        for s_leg in ['r_leg', 'l_leg']:
            if self.control_dimension == 3:
                self.brain_command[s_leg]['theta_tgt_f'] = cp[s_leg]['theta_tgt_f']
                sign_frontral = 1 if s_leg is 'r_leg' else -1
                alpha_tgt_global_frontal = cp[s_leg]['c0_f'] + sign_frontral*cp[s_leg]['cv_f']*s_b['d_pos'][1]
                theta_f = sign_frontral*s_b['theta'][0]
                self.brain_command[s_leg]['alpha_tgt_f'] = alpha_tgt_global_frontal - theta_f

            self.brain_command[s_leg]['theta_tgt'] = cp[s_leg]['theta_tgt']

            alpha_tgt_global = cp[s_leg]['c0'] - cp[s_leg]['cv']*s_b['d_pos'][0]
            self.brain_command[s_leg]['alpha_tgt'] = alpha_tgt_global - s_b['theta'][1]
            self.brain_command[s_leg]['alpha_delta'] = cp[s_leg]['alpha_delta']
            self.brain_command[s_leg]['knee_sw_tgt'] = cp[s_leg]['knee_sw_tgt']
            self.brain_command[s_leg]['knee_tgt'] = cp[s_leg]['knee_tgt']
            self.brain_command[s_leg]['knee_off_st'] = cp[s_leg]['knee_off_st']
            self.brain_command[s_leg]['ankle_tgt'] = cp[s_leg]['ankle_tgt']
            # alpha = hip - 0.5*knee
            self.brain_command[s_leg]['hip_tgt'] = \
                self.brain_command[s_leg]['alpha_tgt'] + 0.5*self.brain_command[s_leg]['knee_tgt']
            
        # select which leg to swing
        self.brain_command['r_leg']['swing_init'] = 0
        self.brain_command['l_leg']['swing_init'] = 0
        if sensor_data['r_leg']['contact_ipsi'] and sensor_data['l_leg']['contact_ipsi']:
            r_delta_alpha = sensor_data['r_leg']['alpha'] - self.brain_command['r_leg']['alpha_tgt']
            l_delta_alpha = sensor_data['l_leg']['alpha'] - self.brain_command['l_leg']['alpha_tgt']
            if r_delta_alpha > l_delta_alpha:
                self.brain_command['r_leg']['swing_init'] = 1
            else:
                self.brain_command['l_leg']['swing_init'] = 1
        
        #Shakiba
        for s_leg in ['r_leg', 'l_leg']:
            self.templeg = s_leg
            self.ph0i = Phase['r_leg']  if s_leg is 'r_leg' else Phase['l_leg']    # ipsilateral leg initial phase
            self.ph0c = Phase['l_leg']  if s_leg is 'r_leg' else Phase['r_leg']    # contralateral leg initial phase
            
            '''
            dPhase = (self.omega-1.7*np.sin(self.ph0i-self.ph0c-np.pi))    
           
            self.Phase[s_leg] = (dPhase*self.TIMESTEP + self.ph0i)%(2*np.pi)
            
            #if self.brain_command[s_leg]['swing_init']: # ( I was resetting the phase several times during swing, This was corrected after W21, the correct phase-resetting works on some models like W19, but not all like w16)
                #self.Phase[s_leg] = self.phi_Off #+ dPhase*self.TIMESTEP
            if self.in_contact[s_leg] and not sensor_data[s_leg]['contact_ipsi']:
                self.Phase[s_leg] = self.phi_Off + dPhase*self.TIMESTEP
                
            if not self.in_contact[s_leg] and sensor_data[s_leg]['contact_ipsi']:
            	self.Phase[s_leg] = self.phi_Con #+ dPhase*self.TIMESTEP
            	
            '''
                # Aoi [2010] Oscillator and Resetting (implemented the delta function on 23/10/2021 after W20)
            if self.in_contact[s_leg] and not sensor_data[s_leg]['contact_ipsi']:
                self.cutan['t_Off'][s_leg] = t_c
                #print('swing {}'.format(s_leg), 'time:',  t_c, 'phase:', self.ph0i/np.pi)
                #print('swing {}:'.format(s_leg), self.ph0i-self.phi_Off)
                
            if not self.in_contact[s_leg] and sensor_data[s_leg]['contact_ipsi']:
            	self.cutan['t_Con'][s_leg] = t_c
            	#print('contact {}'.format(s_leg), 'time:', t_c, 'phase:', self.ph0i/np.pi)
            	#print('contact {}:'.format(s_leg), self.ph0i-self.phi_Con)

            self.Phase[s_leg],t_next = self.Runge_Kutta4(self.ph0i, t_c, self.TIMESTEP)
            #'''

            CPG = np.zeros(5)
            
            s_p = 10    #smoothing parameter
            
            CPG[0] = 0.5*(np.tanh(s_p*(Phase[s_leg]-phi[0])))+0.5 if Phase[s_leg]< phi[0] + dphi[0]/2 else (0.5-0.5*(np.tanh(s_p*(Phase[s_leg]-phi[0]-dphi[0]))))
            CPG[1] = 0.5*(np.tanh(s_p*(Phase[s_leg]-phi[1])))+0.5 if Phase[s_leg]< phi[1] + dphi[1]/2 else (0.5-0.5*(np.tanh(s_p*(Phase[s_leg]-phi[1]-dphi[1]))))
            CPG[2] = 0.5*(np.tanh(s_p*(Phase[s_leg]-phi[2])))+0.5 if Phase[s_leg]< phi[2] + dphi[2]/2 else (0.5-0.5*(np.tanh(s_p*(Phase[s_leg]-phi[2]-dphi[2]))))
            CPG[3] = 0.5*(np.tanh(s_p*(Phase[s_leg]-phi[3])))+0.5 if Phase[s_leg]< phi[3] + dphi[3]/2 else (0.5-0.5*(np.tanh(s_p*(Phase[s_leg]-phi[3]-dphi[3]))))
            CPG[4] = 0.5*(np.tanh(s_p*(Phase[s_leg]-phi[4])))+0.5 if Phase[s_leg]< phi[4] + dphi[4]/2 else (0.5-0.5*(np.tanh(s_p*(Phase[s_leg]-phi[4]-dphi[4]))))
            
            
            for m in muscle:
                self.stim0_ph[s_leg][m]=np.dot(w[m],CPG)

        #print('t_Con_r_leg:', self.cutan['t_Con']['r_leg'])
        #
# -----------------------------------------------------------------------------------------------------------------
    # _spinal_control uses sensory data(without delay?!?!) to update the phase(module) and based on that the stim
    def _spinal_control(self, sensor_data):
        for s_leg in ['r_leg', 'l_leg']:
            self._update_spinal_control_phase(s_leg, sensor_data)
            self.stim[s_leg] = self.spinal_control_leg(s_leg, sensor_data)

# -----------------------------------------------------------------------------------------------------------------
    # _update_spinal_control_phase useing the brain-command and sensory data to update the phase(module)     
    def _update_spinal_control_phase(self, s_leg, sensor_data):
        s_l = sensor_data[s_leg]

        alpha_tgt = self.brain_command[s_leg]['alpha_tgt']
        alpha_delta = self.brain_command[s_leg]['alpha_delta']
        knee_sw_tgt = self.brain_command[s_leg]['knee_sw_tgt']

        # when foot touches ground
        if not self.in_contact[s_leg] and s_l['contact_ipsi']:
            # initiate stance control
            self.spinal_control_phase[s_leg]['ph_st'] = 1
            # swing control off
            self.spinal_control_phase[s_leg]['ph_sw'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_flex_k'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_hold_k'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_stop_l'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_hold_l'] = 0

        # during stance control
        if self.spinal_control_phase[s_leg]['ph_st']:
            # contra-leg in swing (single stance phase)
            self.spinal_control_phase[s_leg]['ph_st_csw'] = not s_l['contact_contra']
            # initiate swing
            self.spinal_control_phase[s_leg]['ph_st_sw0'] = self.brain_command[s_leg]['swing_init']
            # do not initiate swing
            self.spinal_control_phase[s_leg]['ph_st_st'] = not self.spinal_control_phase[s_leg]['ph_st_sw0']

        # when foot loses contact
        if self.in_contact[s_leg] and not s_l['contact_ipsi']:
            # stance control off
            self.spinal_control_phase[s_leg]['ph_st'] = 0
            self.spinal_control_phase[s_leg]['ph_st_csw'] = 0
            self.spinal_control_phase[s_leg]['ph_st_sw0'] = 0
            self.spinal_control_phase[s_leg]['ph_st_st'] = 0
            # initiate swing control
            self.spinal_control_phase[s_leg]['ph_sw'] = 1
            # flex knee
            self.spinal_control_phase[s_leg]['ph_sw_flex_k'] = 1

        # during swing control
        if self.spinal_control_phase[s_leg]['ph_sw']:
            if self.spinal_control_phase[s_leg]['ph_sw_flex_k']:
                if s_l['phi_knee'] < knee_sw_tgt: # knee flexed
                    self.spinal_control_phase[s_leg]['ph_sw_flex_k'] = 0
                    # hold knee
                    self.spinal_control_phase[s_leg]['ph_sw_hold_k'] = 1
            else:
                if self.spinal_control_phase[s_leg]['ph_sw_hold_k']:
                    if s_l['alpha'] < alpha_tgt: # leg swung enough
                        self.spinal_control_phase[s_leg]['ph_sw_hold_k'] = 0
                if s_l['alpha'] < alpha_tgt + alpha_delta: # leg swung enough
                    # stop leg
                    self.spinal_control_phase[s_leg]['ph_sw_stop_l'] = 1
                if self.spinal_control_phase[s_leg]['ph_sw_stop_l'] \
                    and s_l['dalpha'] > 0: # leg started to retract
                    # hold leg
                    self.spinal_control_phase[s_leg]['ph_sw_hold_l'] = 1

        self.in_contact[s_leg] = s_l['contact_ipsi']

# -----------------------------------------------------------------------------------------------------------------
    # spinal_control_leg uses sensory data(without delay?!?!) to update the stim
    def spinal_control_leg(self, s_leg, sensor_data):
        s_l = sensor_data[s_leg]
        s_b = sensor_data['body']
        cp = self.cp[s_leg]

        ph_st = self.spinal_control_phase[s_leg]['ph_st']
        ph_st_csw = self.spinal_control_phase[s_leg]['ph_st_csw']
        ph_st_sw0 = self.spinal_control_phase[s_leg]['ph_st_sw0']
        ph_st_st = self.spinal_control_phase[s_leg]['ph_st_st']
        ph_sw = self.spinal_control_phase[s_leg]['ph_sw']
        ph_sw_flex_k = self.spinal_control_phase[s_leg]['ph_sw_flex_k']
        ph_sw_hold_k = self.spinal_control_phase[s_leg]['ph_sw_hold_k']
        ph_sw_stop_l = self.spinal_control_phase[s_leg]['ph_sw_stop_l']
        ph_sw_hold_l = self.spinal_control_phase[s_leg]['ph_sw_hold_l']

        theta = s_b['theta'][1]
        dtheta = s_b['dtheta'][1]

        sign_frontral = 1 if s_leg is 'r_leg' else -1
        theta_f = sign_frontral*s_b['theta'][0]
        dtheta_f = sign_frontral*s_b['dtheta'][0]

        theta_tgt = self.brain_command[s_leg]['theta_tgt']
        alpha_tgt = self.brain_command[s_leg]['alpha_tgt']
        alpha_delta = self.brain_command[s_leg]['alpha_delta']
        hip_tgt = self.brain_command[s_leg]['hip_tgt']
        knee_tgt = self.brain_command[s_leg]['knee_tgt']
        knee_sw_tgt = self.brain_command[s_leg]['knee_sw_tgt']
        knee_off_st = self.brain_command[s_leg]['knee_off_st']
        ankle_tgt = self.brain_command[s_leg]['ankle_tgt']

        stim = {}
        pre_stim = 0.01

        #Shakiba
        stim0_ph= self.stim0_ph[s_leg]
        wFF = 0   # feedforward weight
        wFB = 1   # feedback weight (set the FB weights to some value other than 1, so that the optimization does not choose 0 for values w)
        
        stim0_ph.update((key, pre_stim + np.array(value)*wFF) for key, value in stim0_ph.items())

        #
        
        if self.control_dimension == 3:
            theta_tgt_f = self.brain_command[s_leg]['theta_tgt_f']
            alpha_tgt_f = self.brain_command[s_leg]['alpha_tgt_f']

            S_HAB_3 = ph_st*s_l['load_ipsi']*np.maximum(
                - cp['HAB_3_PG']*(theta_f-theta_tgt_f)
                - cp['HAB_3_DG']*dtheta_f
                , 0)
            S_HAB_6 = (ph_st_sw0*s_l['load_contra'] + ph_sw)*np.maximum(
                cp['HAB_6_PG']*(s_l['alpha_f'] - alpha_tgt_f)
                , 0)
            #stim['HAB'] = S_HAB_3 + S_HAB_6
            stim['HAB'] = pre_stim + (S_HAB_3 + S_HAB_6)
            
            S_HAD_3 = ph_st*s_l['load_ipsi']*np.maximum(
                cp['HAD_3_PG']*(theta_f-theta_tgt_f)
                + cp['HAD_3_DG']*dtheta_f
                , 0)
            S_HAD_6 = (ph_st_sw0*s_l['load_contra'] + ph_sw)*np.maximum(
                - cp['HAD_6_PG']*(s_l['alpha_f'] - alpha_tgt_f)
                , 0)
            #stim['HAD'] = S_HAD_3 + S_HAD_6
            stim['HAD'] = pre_stim + (S_HAD_3 + S_HAD_6)

        S_HFL_3 = ph_st*s_l['load_ipsi']*np.maximum(
            - cp['HFL_3_PG']*(theta-theta_tgt)
            - cp['HFL_3_DG']*dtheta
            , 0)
        S_HFL_6 = (ph_st_sw0*s_l['load_contra'] + ph_sw)*np.maximum(
            cp['HFL_6_PG']*(s_l['alpha']-alpha_tgt)
            + cp['HFL_6_DG']*s_l['dalpha']
            , 0)
        S_HFL_10 = ph_sw_hold_l*np.maximum(
            cp['HFL_10_PG']*(s_l['phi_hip'] - hip_tgt)
            , 0)
        #stim['HFL'] = pre_stim + S_HFL_3 + S_HFL_6 + S_HFL_10
        #stim['HFL'] = stim0_ph['HFL']+ (S_HFL_3 + S_HFL_6 + S_HFL_10)*wFB
        stim['HFL'] = stim0_ph['HFL']+ ((S_HFL_3/wFB) + S_HFL_6 + S_HFL_10)*wFB

        S_GLU_3 = ph_st*s_l['load_ipsi']*np.maximum(
            cp['GLU_3_PG']*(theta-theta_tgt)
            + cp['GLU_3_DG']*dtheta
            , 0)
        S_GLU_6 = (ph_st_sw0*s_l['load_contra'] + ph_sw)*np.maximum(
            - cp['GLU_6_PG']*(s_l['alpha']-alpha_tgt)
            - cp['GLU_6_DG']*s_l['dalpha']
            , 0)
        S_GLU_10 = ph_sw_hold_l*np.maximum(
            - cp['GLU_10_PG']*(s_l['phi_hip'] - hip_tgt)
            , 0)
        #stim['GLU'] = pre_stim + S_GLU_3 + S_GLU_6 + S_GLU_10
        #stim['GLU'] = stim0_ph['GLU'] + (S_GLU_3 + S_GLU_6 + S_GLU_10)*wFB
        stim['GLU'] = stim0_ph['GLU']+ ((S_GLU_3/wFB) + S_GLU_6 + S_GLU_10)*wFB

        S_HAM_3 = cp['HAM_3_GLU']*S_GLU_3
        S_HAM_9 = ph_sw_stop_l*np.maximum(
            - cp['HAM_9_PG']*(s_l['alpha'] - (alpha_tgt + alpha_delta))
            , 0)
        #stim['HAM'] = pre_stim + S_HAM_3 + S_HAM_9
        stim['HAM'] = stim0_ph['HAM']+ (S_HAM_3 + S_HAM_9)*wFB

        S_RF_1 = (ph_st_st + ph_st_sw0*(1-s_l['load_contra']))*np.maximum(
            cp['RF_1_FG']*s_l['F_RF']
            , 0)
        S_RF_8 = ph_sw_hold_k*np.maximum(
            - cp['RF_8_DG_knee']*s_l['dphi_knee']
            , 0)
        #stim['RF'] = pre_stim + S_RF_1 + S_RF_8
        stim['RF'] = stim0_ph['RF'] + (S_RF_1 + S_RF_8)*wFB

        S_VAS_1 = (ph_st_st + ph_st_sw0*(1-s_l['load_contra']))*np.maximum(
            cp['VAS_1_FG']*s_l['F_VAS']
            , 0)
        S_VAS_2 = -(ph_st_st + ph_st_sw0*(1-s_l['load_contra']))*np.maximum(
            cp['VAS_2_PG']*(s_l['phi_knee'] - knee_off_st)
            , 0)
        S_VAS_10 = ph_sw_hold_l*np.maximum(
            - cp['VAS_10_PG']*(s_l['phi_knee'] - knee_tgt)
            , 0)
        #stim['VAS'] = pre_stim + S_VAS_1 + S_VAS_2 + S_VAS_10
        stim['VAS'] = stim0_ph['VAS']+ (S_VAS_1 + S_VAS_2 + S_VAS_10)*wFB

        S_BFSH_2 = (ph_st_st + ph_st_sw0*(1-s_l['load_contra']))*np.maximum(
            cp['BFSH_2_PG']*(s_l['phi_knee'] - knee_off_st)
            , 0)
        S_BFSH_7 = (ph_st_sw0*(s_l['load_contra']) + ph_sw_flex_k)*np.maximum(
            - cp['BFSH_7_DG_alpha']*s_l['dalpha']
            + cp['BFSH_7_PG']*(s_l['phi_knee'] - knee_sw_tgt)
            , 0)
        S_BFSH_8 = ph_sw_hold_k*np.maximum(
            cp['BFSH_8_DG']*(s_l['dphi_knee'] - knee_off_st)
            *cp['BFSH_8_PG']*(s_l['alpha'] - alpha_tgt)
            , 0)
        S_BFSH_9 = np.maximum(
            cp['BFSH_9_G_HAM']*(S_HAM_9 - cp['BFSH_9_HAM0'])
            , 0)
        S_BFSH_10 = ph_sw_hold_l*np.maximum(
            cp['BFSH_10_PG']*(s_l['phi_knee'] - knee_tgt)
            , 0)
        #stim['BFSH'] = pre_stim + S_BFSH_2 + S_BFSH_7 + S_BFSH_8 + S_BFSH_9 + S_BFSH_10
        stim['BFSH'] = stim0_ph['BFSH'] + (S_BFSH_2 + S_BFSH_7 + S_BFSH_8 + S_BFSH_9 + S_BFSH_10)*wFB

        S_GAS_2 = ph_st*np.maximum(
            cp['GAS_2_FG']*s_l['F_GAS']
            , 0)
        #stim['GAS'] = pre_stim + S_GAS_2
        stim['GAS'] = stim0_ph['GAS'] + (S_GAS_2)*wFB
        
        S_SOL_1 = ph_st*np.maximum(
            cp['SOL_1_FG']*s_l['F_SOL']
            , 0)
        #stim['SOL'] = pre_stim + S_SOL_1
        stim['SOL'] = stim0_ph['SOL'] + (S_SOL_1)*wFB

        S_TA_5 = np.maximum(
            cp['TA_5_PG']*(s_l['phi_ankle'] - ankle_tgt)
            , 0)
        S_TA_5_st = -ph_st*np.maximum(
            cp['TA_5_G_SOL']*S_SOL_1
            , 0)
        #stim['TA'] = pre_stim + S_TA_5 + S_TA_5_st
        stim['TA'] = stim0_ph['TA'] + (S_TA_5 + S_TA_5_st)*wFB

        for muscle in stim:
            stim[muscle] = np.clip(stim[muscle], 0.01, 1.0)

        return stim
# -----------------------------------------------------------------------------------------------------------------
    def delta(self, t_c):
        a = 1e-2
        y = (np.exp(-np.power(t_c/a, 2)))
        return y
# -----------------------------------------------------------------------------------------------------------------
    def dPhasedt(self, z,t):
    	s_leg = self.templeg
    	dz = (self.omega-1.7*np.sin(z-self.ph0c-np.pi)) - (z-self.phi_Off)* self.delta(t-self.cutan['t_Off'][s_leg]-0.05) - (z-self.phi_Con)* self.delta(t-self.cutan['t_Con'][s_leg]-0.05)
    	return dz
        
# -----------------------------------------------------------------------------------------------------------------
    def  Runge_Kutta4(self, z0, t0, dt):
        k1 = self.dPhasedt(z0, t0)*dt
        k2 = self.dPhasedt(z0 + 0.5*k1, t0 + 0.5*dt)*dt
        k3 = self.dPhasedt(z0 + 0.5*k2, t0 + 0.5*dt)*dt
        k4 = self.dPhasedt(z0 + k3, t0 + dt)*dt

        z = (z0 + (k1 + 2*k2 + 2*k3 + k4)/6)%(2*np.pi)
        t = t0 + dt

        return z, t
