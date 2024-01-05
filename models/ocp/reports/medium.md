| pipeline                      | language | accuracy           | params                                                                      | size (MB) |
|-------------------------------|----------|--------------------|-----------------------------------------------------------------------------|-----------|
| media_ocp_cv2_medium          | en       | 0.8683117027695156 | {'penalty': None, 'l1_ratio': 0.5, 'early_stopping': False, 'alpha': 0.01}  | 54.841    |
| media_ocp_cv2_kw_medium *     | en       | 0.9054523222627585 | {'penalty': None, 'l1_ratio': 0.3, 'early_stopping': False, 'alpha': 0.05}  | 0.186     |
| playback_ocp_cv2_kw_medium ** | en       | 0.9491247659281933 | {'penalty': None, 'l1_ratio': 0.5, 'early_stopping': False, 'alpha': 0.07}  | 0.021     |
| binary_ocp_cv2_kw_medium ***  | en       | 0.9931496867494176 | {'penalty': None, 'l1_ratio': 0.15, 'early_stopping': False, 'alpha': 0.07} | 0.01      |

`*` requires `media_ocp_cv2_medium`
`**` requires `playback_ocp_cv2_small`
`***` requires `binary_ocp_cv2_small`
