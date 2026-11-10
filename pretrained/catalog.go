package pretrained

import (
	_ "embed"
	"encoding/json"
)

// Preset identifies a pinned GGUF compatibility target. Files are not bundled
// or downloaded automatically. Upstream weights retain their own licenses.
type Preset struct {
	Name         string `json:"name"`
	Architecture string `json:"architecture"`
	Family       string `json:"family"`
	Repository   string `json:"repository"`
	Filename     string `json:"filename"`
	Revision     string `json:"revision"`
	SHA256       string `json:"sha256"`
	ToolMode     string `json:"tool_mode"`
}

//go:embed testdata/compatibility.json
var compatibilityManifest []byte

// Models lists pinned reference checkpoints for the supported architecture families.
func Models() []Preset {
	var result []Preset
	if err := json.Unmarshal(compatibilityManifest, &result); err != nil {
		panic(err)
	}
	return result
}
