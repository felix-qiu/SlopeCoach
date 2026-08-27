import SwiftUI

struct ResultView: View {
    var body: some View {
        PlaceholderFeatureView(
            title: "Analysis Result",
            systemImage: "chart.xyaxis.line"
        )
        .navigationTitle("Result")
        .navigationBarTitleDisplayMode(.inline)
    }
}
