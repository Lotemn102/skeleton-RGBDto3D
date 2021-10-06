import json

def generate_frames_json():
        save_path = 'frames_sync.json'
        all_data = []

        for i in range(1, 16):
                sub_name = 'Sub00' + str(i) if i < 10 else 'Sub0' + str(i)

                data = {
                        'sub_name' : sub_name,
                        'Squat' :  {
                                'Back': 0,
                                'Front' : 0,
                                'Side' : 0,
                                'Vicon' : 0
                        },

                        'Stand' :  {
                                'Back': 0,
                                'Front' : 0,
                                'Side' : 0,
                                'Vicon' : 0
                        },

                        'Tight' :  {
                                'Back': 0,
                                'Front' : 0,
                                'Side' : 0,
                                'Vicon' : 0
                        },

                        'Left' :  {
                                'Back': 0,
                                'Front' : 0,
                                'Side' : 0,
                                'Vicon' : 0
                        },

                        'Right' :  {
                                'Back': 0,
                                'Front' : 0,
                                'Side' : 0,
                                'Vicon' : 0
                        },
                    }

                all_data.append(data)

        json_data = json.dumps(all_data)
        json_file = open(save_path, "w")
        json_file.write(json_data)
        json_file.close()

