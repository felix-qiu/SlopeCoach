import SwiftUI

struct PlaceholderFeatureView: View {
    let title: String
    let systemImage: String

    var body: some View {
        ZStack {
            Color.slopeBackground
                .ignoresSafeArea()

            ContentUnavailableView(title, systemImage: systemImage)
        }
    }
}
