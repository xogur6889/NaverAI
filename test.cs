using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Linq;

[Serializable]
public class Point
{
    public float x;
    public float y;
}

[Serializable]
public class DynamicObject
{
    public int id;
    public int t = 0; // 0 : car, 1: bus
    public float x;
    public float y;
    public float w;
    public float h;
    public float l;
}

[Serializable]
public class Line
{
    public int t = 0; // 0: 실선, 1: 점선
    public int c = 0; // 0: 흰색, 1: 황색, 2: 청색
    public List<Point> points = new List<Point>();
}

[Serializable]
public class Way
{
    public List<Point> points = new List<Point>();
}



[Serializable]
public class Frame
{
    public float x;
    public float y;
    public List<Line> lines = new List<Line>();
    public List<Way> driveways = new List<Way>();
    public List<Way> sidewalks = new List<Way>();
    public List<DynamicObject> others = new List<DynamicObject>();
    public List<DynamicObject> signals = new List<DynamicObject>();
}

[Serializable]
public class Data
{
    public List<Frame> frames = new List<Frame>();
    public float img_width;
    public float img_height;
    public float pitch;
}

[Serializable]
public class WorldPosResult
{
    public List<Frame> frames = new List<Frame>();
}

public class FrameOther
{
    public int idx;
    public List<GameObject> others = new List<GameObject>();
}

public class ConvertedWorldPosInfo
{
    public int id;
    public int t = 0; // 0 : car, 1: bus
    public float w;
    public float h;
    public float l;
    public Transform tr;
}



public class LineAnchor
{
    public int count = 0;
    public float x;
}

public class test : MonoBehaviour
{
    //public Image team2_result_view;
    public Transform ego_car;
    public Transform road_parent;
    public LineRenderer lineRenderer;
    public LineRenderer line_dotRenderer;
    public Transform car;
    public Transform bus;
    public Transform signal;
    public GameObject road;
    public GameObject bg;
    public Camera cam;
    public Transform cam_parent;
    public Transform python_output_parent;
    public TMP_Text python_output;
    public Transform visual_json_button_parent;
    public Button visual_json_button;
    public Button make_json_button;
    public TMP_InputField video_name_field;
    public float cam_height;

    public GameObject Lidar_prefab;
    GameObject LidarRenderer;
    Mesh mesh;
    MeshFilter mf;
    LineRenderer ego_trace_line;

    //public GameObject trace;
    //public GameObject[] other_objs;
    private Data data = new Data();
    private Sprite[] sprites;
    private List<FrameOther> frameOthers;
    int frame_idx = 0;
    float play_speed = 1f;
    bool is_selected = false;
    Quaternion dest_rotation;
    List<LineRenderer> bef_lines = new List<LineRenderer>();
    Color[] line_colors = { Color.white, Color.yellow, Color.blue };
    Texture2D tex = null;
    WorldPosResult world_pos_result;
    string main_path = "C:/Users/na/Desktop/Pytorch_Generalized_3D_Lane_Detection-master/";
    private static System.Text.StringBuilder line_output = new System.Text.StringBuilder();
    List<string> python_outputs = new List<string>();

    private void Awake()
    {
        // 그리기위한 객체 생성
        LidarRenderer = Instantiate(Lidar_prefab);
        mf = LidarRenderer.GetComponent<MeshFilter>();
        mesh = new Mesh
        {
            // Use 32 bit integer values for the mesh, allows for stupid amount of vertices (2,147,483,647 I think?)
            indexFormat = UnityEngine.Rendering.IndexFormat.UInt32
        };

        tex = new Texture2D(2, 2);

        foreach (string json_path in Directory.GetFiles(main_path, "*json"))
        {
            string json_name = json_path.Split('/').Last();
            json_name = json_name.Split('.').First();
            Button _visual_json_button = Instantiate(visual_json_button, visual_json_button_parent);
            _visual_json_button.GetComponentInChildren<TMP_Text>().text = json_name;
            _visual_json_button.onClick.AddListener(() => Visualize(json_name));
        }
    }

    public void MakeJson()
    {
        try
        {
            if (video_name_field.text == "")
            {
                Debug.Log("뭐라도 입력하세요.");
                return;
            }

            foreach (string json_path in Directory.GetFiles(main_path, "*json"))
            {
                string json_name = json_path.Split('/').Last();
                json_name = json_name.Split('.').First();
                if (video_name_field.text == json_name)
                {
                    Debug.Log("이미 존재하는 JSON 이네요.");
                    return;
                }
            }

            System.Diagnostics.Process line_process = new System.Diagnostics.Process();
            line_process.StartInfo.FileName = main_path + "venv/Scripts/python.exe";
            line_process.StartInfo.Arguments = main_path + "generate_json.py " + video_name_field.text;
            line_process.StartInfo.CreateNoWindow = true;
            line_process.StartInfo.UseShellExecute = false;
            line_process.StartInfo.RedirectStandardOutput = true;
            //line_process.StartInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden;
            line_process.EnableRaisingEvents = true;
            line_process.Exited += new EventHandler(myProcess_Exited);
            line_process.OutputDataReceived += new System.Diagnostics.DataReceivedEventHandler((sender, e) =>
            {
                line_output.Append(e.Data + '\n');
            });
            line_output.Clear();
            make_json_button.interactable = false;
            line_process.Start();
            line_process.BeginOutputReadLine();
        }
        catch (Exception e)
        {
            Debug.LogError("generate_json 망함: " + e.Message);
        }
    }

    private void myProcess_Exited(object sender, System.EventArgs e)
    {
        python_outputs = line_output.ToString().Split('\n').ToList();
    }

    public void Visualize(string video_name)
    {
        frameOthers = new List<FrameOther>();
        sprites = Resources.LoadAll<Sprite>(video_name);

        //if (!Directory.Exists(SAVE_DATA_DIRECTORY)) // 해당 경로가 존재하지 않는다면
        //    Directory.CreateDirectory(SAVE_DATA_DIRECTORY); // 폴더 생성(경로 생성)
        string json_path = main_path + video_name + ".json";
        if (!File.Exists(json_path))
        {
            Debug.Log("파일이 없어용");
            return;
        }
        string loadJson = File.ReadAllText(json_path);
        data = JsonUtility.FromJson<Data>(loadJson);
        Dictionary<int, int> id_freq_pairs = new Dictionary<int, int>();
        Dictionary<int, Vector3> obj_pos_pairs = new Dictionary<int, Vector3>();
        Dictionary<int, Vector3> first_other_world_pos_pairs = new Dictionary<int, Vector3>();
        Dictionary<int, Vector3> last_other_world_pos_pairs = new Dictionary<int, Vector3>();
        world_pos_result = new WorldPosResult();
        foreach (var frame in data.frames)
        {
            foreach (var other in frame.others)
            {
                if (id_freq_pairs.ContainsKey(other.id))
                {
                    ++id_freq_pairs[other.id];
                    Vector3 new_pos = new Vector3(other.x, 0f, other.y);

                    // 팍 튀는거 방지
                    if (Vector3.Distance(obj_pos_pairs[other.id], new_pos) < 3f)
                    {
                        obj_pos_pairs[other.id] = new_pos;
                    }
                    else
                    {
                        other.x = obj_pos_pairs[other.id].x;
                        other.y = obj_pos_pairs[other.id].z;
                    }

                    last_other_world_pos_pairs[other.id] = new Vector3(other.x + frame.x, 0f, other.y + frame.y);
                }
                else
                {
                    id_freq_pairs.Add(other.id, 1);
                    obj_pos_pairs.Add(other.id, new Vector3(other.x, 0f, other.y));
                    // 처음 시작 위치만 담자.
                    first_other_world_pos_pairs.Add(other.id, new Vector3(other.x + frame.x, 0f, other.y + frame.y));
                    // 마지막 위치를 넣을 것.
                    last_other_world_pos_pairs.Add(other.id, new Vector3(other.x + frame.x, 0f, other.y + frame.y));
                }
            }
        }


        Dictionary<int, LineRenderer> valid_id_color_pairs = new Dictionary<int, LineRenderer>();
        foreach (var id_freq_pair in id_freq_pairs)
        {
            // 5번 이상 등장한 객체만..
            if (id_freq_pair.Value > 5)
            {
                valid_id_color_pairs.Add(id_freq_pair.Key, new LineRenderer());
            }
        }

        //// 라인 수를 알 때
        //LineRenderer[] lines = new LineRenderer[4];
        //for (int i = 0; i < lines.Length; ++i)
        //{
        //    lines[i] = Instantiate(lineRenderer, road_parent);
        //    lines[i].positionCount = data.frames.Count;
        //}

        Quaternion bef_rot = ego_car.rotation;
        Transform bef_road_tr = road_parent;
        LineRenderer ego_trace_line = Instantiate(lineRenderer, road_parent);
        ego_trace_line.positionCount = data.frames.Count;
        ego_trace_line.startColor = ego_trace_line.endColor = Color.red;
        Vector3 des_dir = Vector3.forward;

        List<Vector3> color_pos_s = new List<Vector3>();
        List<Color> color_rgb_s = new List<Color>();

        cam.transform.localEulerAngles = new Vector3(data.pitch, 0f, 0f);
        cam.transform.localPosition = new Vector3(0f, cam_height, 0f);
        cam_parent.position = new Vector3(data.frames[0].x, 0f, data.frames[0].y);
        cam_parent.forward = des_dir;

        for (int f_idx = 0; f_idx < data.frames.Count; ++f_idx)
        {
            Frame frame = data.frames[f_idx];
            Transform road_tr = Instantiate(road, road_parent).transform;
            road_tr.position = new Vector3(frame.x, 0f, frame.y);
            ego_trace_line.SetPosition(f_idx, road_tr.position + Vector3.up * 0.2f);
            float next_pos_dist = 0f;

            Frame world_pos_frame = new Frame();
            world_pos_frame.x = frame.x;
            world_pos_frame.y = frame.y;

            if (f_idx + 1 < data.frames.Count)
            {
                Frame next_frame = data.frames[f_idx + 1];
                Vector3 next_pos = new Vector3(next_frame.x, 0f, next_frame.y);
                des_dir = (next_pos - road_tr.position).normalized;
                next_pos_dist = Vector3.Distance(next_pos, road_tr.position);
            }

            cam_parent.position = road_tr.position;
            cam_parent.forward = des_dir;

            foreach (Line line in frame.lines)
            {
                RaycastHit hit;
                // 해상도에 맞게 변형
                Line lane = new Line();
                lane.t = line.t;
                lane.c = line.c;
                foreach (Point point in line.points)
                {
                    if (Physics.Raycast(cam.ScreenPointToRay(new Vector2(point.x / data.img_width * Screen.width, point.y / data.img_height * Screen.height)), out hit))
                    {
                        lane.points.Add(new Point { x = hit.point.x, y = hit.point.z });
                        color_pos_s.Add(hit.point);
                        color_rgb_s.Add(line_colors[line.c]);
                    }
                }
                world_pos_frame.lines.Add(lane);
            }

            foreach (Way way in frame.driveways)
            {
                RaycastHit hit;
                // 해상도에 맞게 변형
                Line lane = new Line();
                foreach (Point point in way.points)
                {
                    if (Physics.Raycast(cam.ScreenPointToRay(new Vector2(point.x / data.img_width * Screen.width, point.y / data.img_height * Screen.height)), out hit))
                    {
                        lane.points.Add(new Point { x = hit.point.x, y = hit.point.z });
                        color_pos_s.Add(hit.point);
                        color_rgb_s.Add(Color.grey);
                    }
                }
                //world_pos_frame.lines.Add(lane);
            }

            foreach (Way way in frame.sidewalks)
            {
                RaycastHit hit;
                // 해상도에 맞게 변형
                Line lane = new Line();
                foreach (Point point in way.points)
                {
                    if (Physics.Raycast(cam.ScreenPointToRay(new Vector2(point.x / data.img_width * Screen.width, point.y / data.img_height * Screen.height)), out hit))
                    {
                        lane.points.Add(new Point { x = hit.point.x, y = hit.point.z });
                        color_pos_s.Add(hit.point);
                        color_rgb_s.Add(Color.green);
                    }
                }
                //world_pos_frame.lines.Add(lane);
            }

            //string filePath = "C:/Users/na/Desktop/Pytorch_Generalized_3D_Lane_Detection-master/data/rgb/" + video_name + "/" + f_idx.ToString("D5") + ".png";

            //if (File.Exists(filePath))
            //{
            //    tex.LoadImage(File.ReadAllBytes(filePath));
            //    for (int i = 383; i < 388; ++i)
            //    {
            //        for (int j = 0; j < img_width; ++j)
            //        {
            //            RaycastHit hit;
            //            // 해상도에 맞게 변형
            //            if (Physics.Raycast(cam.ScreenPointToRay(new Vector2(j / img_width * Screen.width, i / img_height * Screen.height)), out hit))
            //            {
            //                color_pos_s.Add(hit.point);
            //                color_rgb_s.Add(tex.GetPixel(j, i));
            //            }
            //        }
            //    }
            //}

            List<ConvertedWorldPosInfo> otherCWPIs = new List<ConvertedWorldPosInfo>();
            FrameOther fo = new FrameOther();
            fo.idx = f_idx;
            for (int i = 0; i < frame.others.Count; ++i)
            {
                DynamicObject dynamic_obj = frame.others[i];
                int other_id = dynamic_obj.id;
                if (valid_id_color_pairs.ContainsKey(other_id))
                {
                    Transform otherInst = null;
                    if (dynamic_obj.t == 0 && Vector2.Distance(Vector2.zero, new Vector2(dynamic_obj.x, dynamic_obj.y)) > 5f) // 차
                    {
                        otherInst = Instantiate(car, road_tr);
                        //otherInst.GetComponentInChildren<TextMeshProUGUI>().text = other_id.ToString("D3");
                        otherInst.GetComponentInChildren<TextMeshProUGUI>().text = other_id.ToString();
                        //otherInst.GetComponentInChildren<TextMeshProUGUI>().color = valid_id_color_pairs[other_id];
                        otherInst.GetComponentInChildren<TextMeshProUGUI>().color = Color.yellow;
                        //otherInst.GetComponentInChildren<TextMeshProUGUI>().color =
                        //foreach (var mr in otherInst.GetComponentsInChildren<MeshRenderer>())
                        //{
                        //    mr.material.color = valid_id_color_pairs[other_id];
                        //}
                        Vector3 re_pos = new Vector3(dynamic_obj.x, 1f, dynamic_obj.y);
                        otherInst.localPosition = re_pos;
                        otherInst.forward = (last_other_world_pos_pairs[other_id] - first_other_world_pos_pairs[other_id]).normalized;
                        fo.others.Add(otherInst.gameObject);
                    }
                    else if (dynamic_obj.t == 1 && Vector2.Distance(Vector2.zero, new Vector2(dynamic_obj.x, dynamic_obj.y)) > 5f) // 버스
                    {
                        otherInst = Instantiate(bus, road_tr);
                        otherInst.GetComponentInChildren<TextMeshProUGUI>().text = other_id.ToString();
                        //otherInst.GetComponentInChildren<TextMeshProUGUI>().color = valid_id_color_pairs[other_id];
                        otherInst.GetComponentInChildren<TextMeshProUGUI>().color = Color.yellow;
                        Vector3 re_pos = new Vector3(dynamic_obj.x, 1f, dynamic_obj.y);
                        otherInst.localPosition = re_pos;
                        otherInst.forward = (last_other_world_pos_pairs[other_id] - first_other_world_pos_pairs[other_id]).normalized;
                        fo.others.Add(otherInst.gameObject);
                    }
                    //else
                    //{
                    //    Debug.Log("아이디 없음.");
                    //}
                    if (otherInst != null)
                    {
                        otherCWPIs.Add(new ConvertedWorldPosInfo
                        {
                            id = dynamic_obj.id,
                            t = dynamic_obj.t,
                            w = dynamic_obj.w,
                            h = dynamic_obj.h,
                            l = dynamic_obj.l,
                            tr = otherInst
                        });
                    }
                }
            }
            frameOthers.Add(fo);

            List<ConvertedWorldPosInfo> signalCWPIs = new List<ConvertedWorldPosInfo>();
            for (int i = 0; i < frame.signals.Count; ++i)
            {
                DynamicObject dynamic_obj = frame.signals[i];
                Transform signalInst = Instantiate(signal, road_tr);
                Vector3 re_pos = new Vector3(dynamic_obj.x, 1f, dynamic_obj.y);
                signalInst.localPosition = re_pos;
                signalCWPIs.Add(new ConvertedWorldPosInfo
                {
                    id =
                    dynamic_obj.id,
                    t = dynamic_obj.t,
                    w = dynamic_obj.w,
                    h = dynamic_obj.h,
                    l = dynamic_obj.l,
                    tr = signalInst
                });
            }

            road_tr.rotation = Quaternion.Lerp(bef_rot, Quaternion.LookRotation(des_dir), Time.fixedDeltaTime); ;
            bef_rot = road_tr.rotation;

            // 돌리고 나서 포지션을 넣어줘야 반영이 될듯하다.
            foreach (ConvertedWorldPosInfo CWPI in otherCWPIs)
            {
                Vector3 other_pos = CWPI.tr.position;
                world_pos_frame.others.Add(new DynamicObject
                {
                    id =
                    CWPI.id,
                    t = CWPI.t,
                    w = CWPI.w,
                    h = CWPI.h,
                    l = CWPI.l,
                    x = other_pos.x,
                    y = other_pos.z
                });
            }

            foreach (ConvertedWorldPosInfo CWPI in signalCWPIs)
            {
                Vector3 other_pos = CWPI.tr.position;
                world_pos_frame.signals.Add(new DynamicObject
                {
                    id =
                    CWPI.id,
                    t = CWPI.t,
                    w = CWPI.w,
                    h = CWPI.h,
                    l = CWPI.l,
                    x = other_pos.x,
                    y = other_pos.z
                });
            }

            world_pos_result.frames.Add(world_pos_frame);
        }

        UpdateMesh(color_pos_s, color_rgb_s);

        foreach (GameObject other in frameOthers[0].others)
        {
            other.SetActive(true);
        }
        is_selected = true;
        dest_rotation = ego_car.rotation;

        bg.SetActive(false);
    }

    // Update is called once per frame
    //void FixedUpdate()
    void Update()
    {
        if (python_outputs.Count > 0)
        {
            foreach (string output in python_outputs)
            {
                if (!String.IsNullOrEmpty(output))
                {
                    Instantiate(python_output, python_output_parent).text = output;
                }
            }
            python_outputs.Clear();
            make_json_button.interactable = true;
            Button _visual_json_button = Instantiate(visual_json_button, visual_json_button_parent);
            TMP_Text visual_json_button_text = _visual_json_button.GetComponentInChildren<TMP_Text>();
            visual_json_button_text.text = video_name_field.text;
            _visual_json_button.onClick.AddListener(() => Visualize(visual_json_button_text.text));
            video_name_field.text = "";

            Visualize(visual_json_button_text.text);

            string json = JsonUtility.ToJson(world_pos_result);
            File.WriteAllText(main_path + "world_pos_result/" + visual_json_button_text.text + ".json", json);

            try
            {
                System.Diagnostics.Process line_process = new System.Diagnostics.Process();
                line_process.StartInfo.FileName = main_path + "venv/Scripts/python.exe";
                line_process.StartInfo.Arguments = main_path + "make_3d_json.py " + visual_json_button_text.text;
                line_process.StartInfo.CreateNoWindow = true;
                line_process.StartInfo.UseShellExecute = false;
                //line_process.EnableRaisingEvents = true;
                //line_process.Exited += new EventHandler(myProcess_Exited);
                line_process.Start();
            }
            catch (Exception e)
            {
                Debug.LogError("make_3d_json 망함: " + e.Message);
            }
            return;
        }

        if (!is_selected) return;
        //if (Input.GetKeyDown(KeyCode.V))
        //{
        //    team2_result_view.gameObject.SetActive(!team2_result_view.gameObject.activeInHierarchy);
        //}

        Frame frame = data.frames[frame_idx];
        // 맨 처음
        if (frame_idx == 0 && ego_car.position == Vector3.zero)
        {
            ScreenCapture.CaptureScreenshot("capture/" + frame_idx + ".jpg");

            ego_car.position = new Vector3(data.frames[0].x, 0f, data.frames[0].y);
            ego_car.forward = Vector3.forward;
        }
        Vector3 dst_pos = new Vector3(frame.x, 0f, frame.y);

        if (Vector3.Distance(dst_pos, ego_car.position) <= Time.deltaTime * play_speed)
        {
            foreach (GameObject other in frameOthers[frame_idx].others)
            {
                other.SetActive(false);
            }
            //Instantiate(trace, ego_car.position + Vector3.up, Quaternion.identity, road_parent);
            ++frame_idx;
            if (frame_idx >= data.frames.Count)
            {
                frame_idx = 0;
                ego_car.position = Vector3.zero;
                ego_car.forward = Vector3.forward;
                is_selected = false;
                bg.SetActive(true);

                for (int i = 0; i < road_parent.childCount; ++i)
                {
                    Destroy(road_parent.GetChild(i).gameObject);
                }
                bef_lines = new List<LineRenderer>();
                return;
            }
            frame = data.frames[frame_idx];
            dst_pos = new Vector3(frame.x, 0f, frame.y);
            //team2_result_view.sprite = sprites[frame_idx];
            foreach (GameObject other in frameOthers[frame_idx].others)
            {
                other.SetActive(true);
            }
            dest_rotation = Quaternion.LookRotation((dst_pos - ego_car.position).normalized);

            ScreenCapture.CaptureScreenshot("capture/" + frame_idx + ".jpg");
        }
        Vector3 delta_step = (dst_pos - ego_car.position).normalized * Time.deltaTime * play_speed;
        delta_step.y = 0f;
        ego_car.rotation = Quaternion.Lerp(ego_car.rotation, dest_rotation, Time.deltaTime);
        ego_car.position += delta_step;
        //cam_parent.position = ego_car.position;
        //cam_parent.rotation = ego_car.rotation;
        //if (frame_idx < data.frames.Count - 1)
        //{
        //    Frame next_frame = data.frames[frame_idx + 1];
        //    ego_car.rotation = Quaternion.Slerp(ego_car.rotation, Quaternion.LookRotation(new Vector3(next_frame.x - ego_car.position.x, 0f, next_frame.y - ego_car.position.z).normalized), Time.deltaTime * play_speed);
        //}

        //for (int i = 0; i < other_objs.Length; ++i)
        //{
        //    if (i < frame.others.Count)
        //    {
        //        other_objs[i].SetActive(true);
        //        Point p = frame.others[i];
        //        Vector3 re_pos = new Vector3(p.x, 0f, p.y);
        //        // 너무 가까이 있는거는 그리지 말자.
        //        if (re_pos.magnitude > 5f)
        //        {
        //            other_objs[i].transform.localPosition = re_pos;
        //        }
        //        else
        //        {
        //            other_objs[i].SetActive(false);
        //        }
        //    }
        //    else
        //    {
        //        other_objs[i].SetActive(false);
        //    }
        //}
    }
    void UpdateMesh(List<Vector3> color_pos_s, List<Color> color_rgb_s)
    {
        if (color_pos_s.Count == 0 || color_rgb_s.Count == 0)
        {
            return;
        }

        mesh.Clear();
        mesh.vertices = color_pos_s.ToArray();
        mesh.colors = color_rgb_s.ToArray();
        int[] indices = new int[color_pos_s.Count];

        for (int i = 0; i < color_pos_s.Count; i++)
        {
            indices[i] = i;
        }

        mesh.SetIndices(indices, MeshTopology.Points, 0);
        mf.mesh = mesh;
    }
}
