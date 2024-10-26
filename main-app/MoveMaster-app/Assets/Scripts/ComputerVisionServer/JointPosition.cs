public class JointPosition
{
    public float x { get; set; }
    public float y { get; set; }
    public bool flag { get; set; }

    public JointPosition(float x, float y, bool flag)
    {
        this.x = x;
        this.y = y;
        this.flag = flag;
    }
}
